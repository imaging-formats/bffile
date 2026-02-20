"""Benchmark comparing old vs new thumbnail generation strategies.

The old strategy (main branch) delegates to Bio-Formats Java ``openThumbBytes``,
which uses AWT image scaling internally.

The new strategy (PR #54) reads the lowest available resolution level directly
and downscales using pure NumPy — avoiding AWT and being much faster for files
that already ship pyramid levels.

Usage::

    # Benchmark all test data files
    python scripts/benchmark_thumbnail.py

    # Benchmark a specific file
    python scripts/benchmark_thumbnail.py path/to/image.tiff

    # Control repetitions
    python scripts/benchmark_thumbnail.py --reps 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bffile import BioFile
from bffile._biofile import (
    THUMB_MAX_READ_SIZE,
    _reshape_image_buffer,
    _resize_thumbnail,
    _thumbnail_target_size,
)


# ---------------------------------------------------------------------------
# Old approach: delegate to Java openThumbBytes (main-branch behaviour)
# ---------------------------------------------------------------------------

def _thumb_java(bf: BioFile, series: int = 0) -> np.ndarray:
    """Original thumbnail via Java openThumbBytes."""
    reader = bf._ensure_java_reader()
    meta = bf.core_metadata(series=series)
    with bf._lock:
        reader.setSeries(series)
        z = meta.shape.z // 2
        idx = reader.getIndex(z, 0, 0)
        java_buffer = memoryview(reader.openThumbBytes(idx))  # type: ignore[attr-defined]
        thumb = np.frombuffer(java_buffer, meta.dtype).copy()
        return _reshape_image_buffer(
            thumb,
            dtype=meta.dtype,
            height=reader.getThumbSizeY(),
            width=reader.getThumbSizeX(),
            rgb=meta.shape.rgb,
            interleaved=meta.is_interleaved,
        )


# ---------------------------------------------------------------------------
# New approach: lowest-resolution read + NumPy downscale (PR #54 behaviour)
# ---------------------------------------------------------------------------

def _thumb_python(bf: BioFile, series: int = 0, max_size: int = 128) -> np.ndarray:
    """New thumbnail using lowest resolution level + NumPy downscaling."""
    return bf.get_thumbnail(series=series, max_size=max_size)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _timeit(fn: Callable[[], Any], reps: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over *reps* calls."""
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def benchmark_file(path: Path, reps: int = 3) -> None:
    print(f"\nFile: {path.name}")
    with BioFile(path) as bf:
        meta = bf.core_metadata()
        n_res = meta.resolution_count
        print(
            f"  Shape: {meta.shape}  |  dtype: {meta.dtype}  |  resolutions: {n_res}"
        )

        # Warm-up (also triggers JVM + memo cache)
        try:
            _thumb_java(bf)
            _thumb_python(bf)
        except Exception as exc:
            print(f"  Skipped (warm-up failed): {exc}")
            return

        mean_j, std_j = _timeit(lambda: _thumb_java(bf), reps)
        mean_p, std_p = _timeit(lambda: _thumb_python(bf), reps)

    speedup = mean_j / mean_p if mean_p > 0 else float("inf")
    print(f"  {'Strategy':<30} {'mean (ms)':>10} {'± std':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Java openThumbBytes (main)':<30} {mean_j:>10.1f} {std_j:>8.1f}")
    print(f"  {'Python NumPy (PR #54)':<30} {mean_p:>10.1f} {std_p:>8.1f}")
    print(f"  Speedup: {speedup:.2f}x  ({'faster' if speedup > 1 else 'slower'})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="File(s) to benchmark (default: all files in tests/data/)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per strategy (default: 3)",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        data_dir = Path(__file__).parent.parent / "tests" / "data"
        if not data_dir.exists() or not any(data_dir.iterdir()):
            print(
                "No test data found.  Run:\n"
                "    python scripts/fetch_test_data.py\n"
                "or provide file paths as arguments."
            )
            sys.exit(1)
        paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix != ".json")

    print(f"Thumbnail benchmark  (reps={args.reps})")
    print("=" * 60)
    for path in paths:
        try:
            benchmark_file(path, reps=args.reps)
        except Exception as exc:
            print(f"\nFile: {path.name}  ->  ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
