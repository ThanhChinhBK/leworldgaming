"""Draw the evaluated controller pipelines without loading experiment records.

From the repository root, with Matplotlib installed:
    python paper/draw_controller_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path(__file__).resolve().parent / "figures"
POSITIONS = (0.05, 1.85, 4.3, 7.05, 9.0)
WIDTHS = (1.45, 2.05, 2.3, 1.55, 0.95)
HEIGHT = 0.56


def box(axis, index: int, y: float, label: str, color: str) -> None:
    x, width = POSITIONS[index], WIDTHS[index]
    axis.add_patch(
        FancyBboxPatch(
            (x, y - HEIGHT / 2),
            width,
            HEIGHT,
            boxstyle="round,pad=0.035,rounding_size=0.06",
            facecolor=color,
            edgecolor="#475569",
            linewidth=0.8,
        )
    )
    axis.text(x + width / 2, y, label, ha="center", va="center", fontsize=8.2)


def forward_arrows(axis, y: float) -> None:
    for index in range(len(POSITIONS) - 1):
        axis.annotate(
            "",
            xy=(POSITIONS[index + 1] - 0.055, y),
            xytext=(POSITIONS[index] + WIDTHS[index] + 0.055, y),
            arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 0.9},
        )


def candidates(axis, y: float, target: int) -> None:
    source_x = POSITIONS[3] + WIDTHS[3] / 2
    target_x = POSITIONS[target] + WIDTHS[target] / 2
    lower_y = y - 0.49
    axis.plot(
        [source_x, source_x, target_x],
        [y - HEIGHT / 2 - 0.04, lower_y, lower_y],
        color="#475569",
        linewidth=0.8,
    )
    axis.annotate(
        "",
        xy=(target_x, y - HEIGHT / 2 - 0.04),
        xytext=(target_x, lower_y),
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 0.8},
    )
    axis.text(
        (source_x + target_x) / 2,
        lower_y,
        "candidate sequences",
        fontsize=7,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.1, 2.4))
    figure.subplots_adjust(left=0.015, right=0.995, top=0.99, bottom=0.035)
    axis.set_xlim(-0.05, 10.05)
    axis.set_ylim(-0.07, 3.35)
    axis.axis("off")

    rows = (
        (
            2.72,
            "(a) LeWM: visual model with online planning",
            "#E7F1FB",
            (
                "RGB frames\n224 x 224",
                "ViT encoder\n+ latent history",
                "Latent predictor\n+ reward / value",
                "Categorical\nCEM",
                "Sampled\naction",
            ),
        ),
        (
            1.58,
            "(b) DreamerV3: learned policy from imagination",
            "#E7F5EE",
            (
                "42-D state\nvector",
                "Encoder + recurrent\nstate update",
                "Current latent\nstate",
                "Learned actor\n(no CEM)",
                "Action",
            ),
        ),
        (
            0.46,
            "(c) PETS: ensemble dynamics with online planning",
            "#FFF1E4",
            (
                "26-D state\nvector",
                "Five-model\nprobabilistic ensemble",
                "Sampled rollouts\n+ HP-change reward",
                "Categorical\nCEM",
                "Mode\naction",
            ),
        ),
    )
    for y, title, color, labels in rows:
        axis.text(0.05, y + 0.42, title, fontsize=9, fontweight="bold", va="center")
        for index, label in enumerate(labels):
            box(axis, index, y, label, color)
        forward_arrows(axis, y)
    candidates(axis, rows[0][0], target=2)
    candidates(axis, rows[2][0], target=1)
    axis.text(
        5.05,
        rows[1][0] - 0.45,
        "Training only: imagined trajectories train the actor and critic.",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#475569",
    )

    OUTPUT.mkdir(exist_ok=True)
    figure.savefig(
        OUTPUT / "controller_diagrams.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(OUTPUT / "controller_diagrams.png", dpi=200)
    plt.close(figure)
    print(f"Rendered controller diagrams in {OUTPUT}")


if __name__ == "__main__":
    main()
