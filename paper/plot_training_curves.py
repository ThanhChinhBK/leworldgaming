#!/usr/bin/env python3
"""Render the archived Dreamer training diagnostics, not benchmark performance.

From the repository root:
    .venv/bin/python paper/plot_training_curves.py
    .venv/bin/python paper/plot_training_curves.py --extract --check

The default renders only the committed CSV; --extract additionally reads the
original stdout log. No checkpoints are deserialized and no training is run.
--check runs in-memory parser/smoothing tests and validates the archived data
and generated PDF/PNG. Matplotlib and NumPy are needed only for rendering.

Provenance: data/dreamer_retrain_incoming_v1.log identifies the incoming-action
run and explicitly saves data/dreamer_checkpoint_incoming_v1.pt at step 63000.
The source has 6341 observations at indices 0..63400 every 10; only the 6301
observations through the evaluated checkpoint are retained. The source has no
resume marker. The printed target 71000 is not an achieved training duration.
The driver logs a zero-based loop index after agent.learn, then periodically
saves that same index as num_steps. Thus the axis is the logged driver index,
not environment frames or an exact count of completed optimizer updates.
Resume restores weights/optimizers and begins again at the saved index; no
resumed segments are spliced into this figure.

Metric definitions (external/dreamerv3-torch/models.py:108-169,183-192 and
networks.py:272-290): kl is the batch/time mean posterior-to-prior KL before
free-nats clipping or dynamics/representation weighting; reward_loss is the
batch/time mean negative reward-head log probability, before loss scaling;
cont_loss is the corresponding binary negative log likelihood of 1-is_terminal.
These are training-batch diagnostics, not held-out losses. Total model loss,
vector reconstruction, actor and critic losses were not printed, so cannot be
recovered from this log. Neither LeWM nor PETS has an archived scalar trace in
data/. All 266 TensorBoard files found there contained only 88-byte headers.

The CSV preserves every printed observation at its original four-decimal
precision, including rounded zeros. No interpolation or downsampling is used.
Thin lines show raw observations; thick lines show the trailing arithmetic
mean of at most 101 observations (1000 driver indices between first and last),
with progressively smaller windows at the beginning. No warmup is discarded.
Reward and continuation use independent symmetric-log axes (linear thresholds
0.01 and 0.0001) so printed zeros are retained without artificial epsilon.
Unlike losses must not be compared as cross-model quality measures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "data/dreamer_retrain_incoming_v1.log"
CHECKPOINT = "data/dreamer_checkpoint_incoming_v1.pt"
STEPS = 63000
KEYS = ("kl", "reward_loss", "cont_loss")
WINDOW = 101
CSV_PATH = ROOT / "paper/figures/dreamer_training_metrics.csv"
FIGURE_PATH = ROOT / "paper/figures/dreamer_training_curves"
ROW = re.compile(r"^\[train_dreamer\] step=\s*(\d+) (.+)$")
VALUE = re.compile(r"([a-z_]+)=([0-9]+\.[0-9]{4})")


def parse_log(text: str, cutoff: int = STEPS) -> list[dict[str, str]]:
    """Reject ambiguous run splices, malformed rows and unproven checkpoint links."""
    if f"ckpt={CHECKPOINT}" not in text:
        raise ValueError("Log does not identify the evaluated checkpoint path")
    marker = f"checkpoint saved -> {CHECKPOINT} (step={cutoff})"
    if marker not in text:
        raise ValueError("Missing checkpoint-save marker at the requested cutoff")
    if "resumed from" in text:
        raise ValueError("Resumed logs require separate provenance review")
    rows = []
    previous = -10
    for line in text.splitlines():
        if not line.startswith("[train_dreamer] step="):
            continue
        match = ROW.fullmatch(line)
        if not match:
            raise ValueError(f"Malformed metric line: {line}")
        step = int(match[1])
        pairs = VALUE.findall(match[2])
        if (
            tuple(key for key, _ in pairs) != KEYS
            or " ".join(f"{key}={value}" for key, value in pairs) != match[2]
        ):
            raise ValueError(f"Unexpected metric fields: {line}")
        if step != previous + 10:
            raise ValueError("Metric indices must start at zero and advance by ten")
        previous = step
        if step <= cutoff:
            rows.append({"step": str(step), **dict(pairs)})
    if not rows or int(rows[-1]["step"]) != cutoff:
        raise ValueError("No metric observation at checkpoint cutoff")
    return rows


def extract() -> None:
    source_bytes = (ROOT / SOURCE).read_bytes()
    rows = parse_log(source_bytes.decode("utf-8"))
    metadata = {
        "source": SOURCE,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "checkpoint": CHECKPOINT,
        "checkpoint_logged_step": STEPS,
        "metric_keys": list(KEYS),
        "step_semantics": "zero-based driver index logged after agent.learn",
        "retained_range": [0, STEPS],
        "sample_interval": 10,
        "observations": len(rows),
        "printed_decimal_places": 4,
        "smoothing": "none in CSV; plot uses trailing mean, window=101, min_count=1",
    }
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as stream:
        stream.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        writer = csv.DictWriter(stream, fieldnames=("step", *KEYS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv() -> tuple[dict, list[dict[str, str]]]:
    with CSV_PATH.open(newline="") as stream:
        line = stream.readline()
        if not line.startswith("# "):
            raise ValueError("Missing CSV provenance")
        metadata = json.loads(line[2:])
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["step", *KEYS]:
            raise ValueError("Unexpected CSV columns")
        rows = list(reader)
    if (
        metadata["source"] != SOURCE
        or metadata["checkpoint"] != CHECKPOINT
        or metadata["checkpoint_logged_step"] != STEPS
        or metadata["metric_keys"] != list(KEYS)
        or metadata["retained_range"] != [0, STEPS]
        or metadata["sample_interval"] != 10
        or metadata["observations"] != len(rows)
        or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_sha256"])
    ):
        raise ValueError("CSV provenance does not match the selected run")
    if [int(row["step"]) for row in rows] != list(range(0, STEPS + 1, 10)):
        raise ValueError("Incomplete or nonmonotonic CSV series")
    for row in rows:
        for key in KEYS:
            if not re.fullmatch(r"[0-9]+\.[0-9]{4}", row[key]):
                raise ValueError("CSV must preserve four-decimal logged values")
            if not math.isfinite(float(row[key])):
                raise ValueError("Nonfinite metric")
    return metadata, rows


def trailing_mean(values: list[float], window: int = WINDOW) -> list[float]:
    if window < 1:
        raise ValueError("Window must be positive")
    total = 0.0
    means = []
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        means.append(total / min(index + 1, window))
    return means


def render(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    steps = [int(row["step"]) / 1000 for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    panels = (
        ("kl", "(a) Latent KL", "Posterior-to-prior KL", "#0072B2", None),
        ("reward_loss", "(b) Reward prediction", "Reward NLL (symlog)", "#D55E00", 0.01),
        ("cont_loss", "(c) Continuation", "Binary NLL (symlog)", "#009E73", 0.0001),
    )
    for axis, (key, title, ylabel, color, threshold) in zip(axes, panels, strict=True):
        values = [float(row[key]) for row in rows]
        axis.plot(steps, values, color=color, alpha=0.25, linewidth=0.4, rasterized=True)
        axis.plot(steps, trailing_mean(values), color=color, linewidth=1.2)
        if threshold is not None:
            axis.set_yscale("symlog", linthresh=threshold, linscale=0.5)
            axis.set_yticks([0, 0.01, 0.1, 1] if key == "reward_loss" else [0, 0.0001, 0.01, 1])
            axis.set_yticklabels(
                ["0", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
                if key == "reward_loss"
                else ["0", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"]
            )
        axis.set_ylim(bottom=0)
        axis.set_xlim(0, 63)
        axis.set_xticks([0, 20, 40, 63])
        axis.axvline(63, color="0.35", linestyle=":", linewidth=0.8)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Logged training step (thousands; zero-based driver index)", y=0.19, fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], color="0.5", alpha=0.4, linewidth=0.8, label="Raw (every 10 steps)"),
            Line2D([], [], color="0.25", linewidth=1.2, label="Trailing mean (101 observations)"),
            Line2D(
                [], [], color="0.35", linestyle=":", linewidth=0.8, label="Evaluated checkpoint"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        ncol=3,
        fontsize=8,
        handlelength=2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.35, wspace=0.48)
    fig.savefig(FIGURE_PATH.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE_PATH.with_suffix(".png"), dpi=220)
    plt.close(fig)


class ParserTests(unittest.TestCase):
    def sample(self) -> str:
        return (
            f"[train_dreamer] data=unused eps=unused ckpt={CHECKPOINT}\n"
            "[train_dreamer] step=    0 kl=1.0000 reward_loss=2.0000 cont_loss=0.0000\n"
            "[train_dreamer] step=   10 kl=1.5000 reward_loss=1.0000 cont_loss=0.0001\n"
            f"[train_dreamer] checkpoint saved -> {CHECKPOINT} (step=10)\n"
            "[train_dreamer] step=   20 kl=2.0000 reward_loss=0.5000 cont_loss=0.0000\n"
        )

    def test_cutoff_and_precision(self) -> None:
        rows = parse_log(self.sample(), cutoff=10)
        self.assertEqual([row["step"] for row in rows], ["0", "10"])
        self.assertEqual(rows[0]["cont_loss"], "0.0000")

    def test_reject_ambiguous_or_malformed_source(self) -> None:
        for text in (
            self.sample().replace("(step=10)", "(step=20)"),
            self.sample().replace("step=   10 kl", "step=    0 kl"),
            self.sample().replace("reward_loss=1.0000", "reward_loss=nan"),
            self.sample().replace(CHECKPOINT, "unrelated.pt"),
            self.sample() + "[train_dreamer] resumed from checkpoint\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_log(text, cutoff=10)

    def test_trailing_mean_keeps_warmup(self) -> None:
        self.assertEqual(trailing_mean([1, 3, 5, 7], 3), [1, 2, 3, 5])
        self.assertEqual(trailing_mean([0, 0, 0]), [0, 0, 0])
        with self.assertRaises(ValueError):
            trailing_mean([1], 0)


def check(metadata: dict, rows: list[dict[str, str]]) -> None:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ParserTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    source = ROOT / SOURCE
    if source.exists():
        source_bytes = source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != metadata["source_sha256"]:
            raise ValueError("Source changed since extraction")
        if rows != parse_log(source_bytes.decode("utf-8")):
            raise ValueError("CSV differs from source observations")
    pdf = FIGURE_PATH.with_suffix(".pdf").read_bytes()
    png = FIGURE_PATH.with_suffix(".png").read_bytes()
    if not pdf.startswith(b"%PDF-") or b"%%EOF" not in pdf[-1024:]:
        raise ValueError("Invalid or incomplete PDF")
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 1000:
        raise ValueError("Invalid PNG")
    import matplotlib.image as mpimg

    image = mpimg.imread(io.BytesIO(png), format="png")
    if image.shape[0] < 450 or image.shape[1] < 1400 or float(image.std()) < 0.01:
        raise ValueError("Rendered figure is undersized or blank")
    print(f"Validated {len(rows)} observations, steps 0..{STEPS}; PDF and PNG readable.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--extract", action="store_true", help="Extract the original stdout log before plotting"
    )
    parser.add_argument(
        "--check", action="store_true", help="Run focused parser, provenance and rendering checks"
    )
    args = parser.parse_args()
    if args.extract:
        extract()
    metadata, rows = load_csv()
    render(rows)
    if args.check:
        check(metadata, rows)
    print(f"Rendered {FIGURE_PATH.relative_to(ROOT)}.{{pdf,png}} from {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
