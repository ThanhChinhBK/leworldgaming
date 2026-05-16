"""Compress raw pixel data in replay HDF5 files after collection.

During data collection, pixels are written uncompressed for speed (~0.2ms/frame
vs ~78ms with LZF). This script re-packs the file with LZF compression on the
pixel dataset, typically achieving 3-5× size reduction.

Usage:
    uv run python scripts/compress_replay.py /path/to/replay.h5
    uv run python scripts/compress_replay.py /path/to/directory/   # all .h5 files
    uv run python scripts/compress_replay.py --all                 # default DATA_DIR
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import h5py

DEFAULT_DATA_DIR = "/media/jeovach/New Volume/leworldgaming"


def compress_file(src: Path) -> None:
    """Re-pack an HDF5 file, adding LZF compression to the pixels dataset."""
    tmp = src.with_suffix(".h5.tmp")
    print(f"Compressing: {src.name}")
    t0 = time.monotonic()
    orig_size = src.stat().st_size / (1024**3)

    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        for name in fin:
            ds = fin[name]
            if isinstance(ds, h5py.Group):
                _copy_group(fin[name], fout.require_group(name))
            elif name == "pixels":
                # Compress pixels with LZF
                shape = ds.shape
                chunk = (64, *shape[1:])
                out_ds = fout.create_dataset(
                    name,
                    shape=shape,
                    dtype=ds.dtype,
                    chunks=chunk,
                    compression="lzf",
                )
                # Copy in batches to avoid loading entire dataset into memory
                batch = 512
                for start in range(0, shape[0], batch):
                    end = min(start + batch, shape[0])
                    out_ds[start:end] = ds[start:end]
            else:
                # Copy other datasets preserving compression
                fout.create_dataset(
                    name,
                    data=ds[()],
                    chunks=ds.chunks,
                    dtype=ds.dtype,
                    compression=ds.compression,
                )

    new_size = tmp.stat().st_size / (1024**3)
    elapsed = time.monotonic() - t0
    ratio = orig_size / max(new_size, 0.001)

    # Replace original with compressed version
    shutil.move(tmp, src)
    print(
        f"  {orig_size:.2f} GB -> {new_size:.2f} GB ({ratio:.1f}x) in {elapsed:.0f}s"
    )


def _copy_group(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    """Recursively copy a group, preserving compression on non-pixel datasets."""
    for name in src_grp:
        item = src_grp[name]
        if isinstance(item, h5py.Group):
            _copy_group(item, dst_grp.require_group(name))
        else:
            dst_grp.create_dataset(
                name,
                data=item[()],
                chunks=item.chunks,
                dtype=item.dtype,
                compression=item.compression or "lzf",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress replay HDF5 pixel data")
    parser.add_argument(
        "paths",
        nargs="*",
        help="HDF5 files or directories to compress (default: DATA_DIR)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Compress all .h5 files in {DEFAULT_DATA_DIR}",
    )
    args = parser.parse_args()

    files: list[Path] = []

    if args.all or not args.paths:
        data_dir = Path(DEFAULT_DATA_DIR)
        files = sorted(data_dir.glob("*.h5"))
    else:
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                files.extend(sorted(path.glob("*.h5")))
            elif path.is_file() and path.suffix == ".h5":
                files.append(path)
            else:
                print(f"Skipping: {p} (not a .h5 file or directory)", file=sys.stderr)

    if not files:
        print("No .h5 files found to compress.")
        return

    print(f"Found {len(files)} file(s) to compress\n")
    total_before = sum(f.stat().st_size for f in files) / (1024**3)

    for f in files:
        compress_file(f)

    total_after = sum(f.stat().st_size for f in files) / (1024**3)
    print(f"\nTotal: {total_before:.2f} GB -> {total_after:.2f} GB")


if __name__ == "__main__":
    main()
