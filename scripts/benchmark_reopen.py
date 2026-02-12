"""Benchmark comparing Bio-Formats file reopen strategies.

Compares three approaches:
1. Keep file open (baseline) — no close/reopen overhead
2. close() + open() — fast handle release/reacquire (reopenFile)
3. Full destroy + Memoizer reopen — complete teardown and rebuild

Usage:
    python scripts/benchmark_reopen.py [path_to_file]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bffile import BioFile


def benchmark_keep_open(path: Path, n_reads: int = 20) -> float:
    """Baseline: keep file open, read planes."""
    with BioFile(path) as bf:
        meta = bf.core_metadata()
        shape = meta.shape
        t0 = time.perf_counter()
        for _ in range(n_reads):
            bf.read_plane(t=0, c=0, z=min(1, shape.z - 1))
        return (time.perf_counter() - t0) / n_reads


def benchmark_suspend_resume(path: Path, n_cycles: int = 20) -> float:
    """Suspend/resume: close() + open() between reads."""
    with BioFile(path) as bf:
        meta = bf.core_metadata()
        shape = meta.shape
        t0 = time.perf_counter()
        for _ in range(n_cycles):
            bf.close()
            bf.open()
            bf.read_plane(t=0, c=0, z=min(1, shape.z - 1))
        return (time.perf_counter() - t0) / n_cycles


def benchmark_full_reopen(path: Path, n_cycles: int = 10) -> float:
    """Full close + Memoizer reopen between reads."""
    # First open to ensure memo file exists
    with BioFile(path, memoize=1) as bf:
        bf.read_plane()

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        with BioFile(path, memoize=1) as bf:
            meta = bf.core_metadata()
            shape = meta.shape
            bf.read_plane(t=0, c=0, z=min(1, shape.z - 1))
    return (time.perf_counter() - t0) / n_cycles


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        data_dir = Path(__file__).parent.parent / "tests" / "data"
        candidates = list(data_dir.glob("*.nd2")) + list(data_dir.glob("*.tiff"))
        if not candidates:
            print("No test files found. Provide a file path as argument.")
            sys.exit(1)
        path = candidates[0]

    print(f"Benchmarking with: {path.name}")
    print(f"{'Strategy':<30} {'Avg time (ms)':>15}")
    print("-" * 47)

    t_open = benchmark_keep_open(path) * 1000
    print(f"{'Keep open (baseline)':<30} {t_open:>15.2f}")

    t_suspend = benchmark_suspend_resume(path) * 1000
    print(f"{'Suspend/resume':<30} {t_suspend:>15.2f}")

    t_reopen = benchmark_full_reopen(path) * 1000
    print(f"{'Full close + Memoizer reopen':<30} {t_reopen:>15.2f}")

    print()
    print(f"Suspend/resume overhead vs baseline: {t_suspend - t_open:.2f} ms")
    print(f"Full reopen overhead vs baseline: {t_reopen - t_open:.2f} ms")
    if t_suspend > 0:
        print(f"Full reopen is {t_reopen / t_suspend:.1f}x slower than suspend/resume")


if __name__ == "__main__":
    main()
