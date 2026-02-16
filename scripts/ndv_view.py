#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bffile[dask,xarray]",
#     "ndv[vispy,pyqt]",
# ]
#
# [tool.uv.sources]
# bffile = { path = "../" }
# ///
"""
View microscopy files using bffile and ndv.

Usage:
    uv run scripts/ndv_view.py <path_to_file> [options]

Examples:
    uv run scripts/ndv_view.py image.nd2
    uv run scripts/ndv_view.py image.czi --series 1 --res 0
    uv run scripts/ndv_view.py large_file.nd2 --dask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ndv  # pyright: ignore[reportMissingImports]

from bffile import BioFile, imread


def main() -> None:
    """Open a microscopy file with bffile and display it with ndv.imshow()."""
    parser = argparse.ArgumentParser(
        description="View files using bffile and ndv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_path", type=Path, help="File to open")
    parser.add_argument(
        "-s",
        "--series",
        type=int,
        default=0,
        help="Which series to load (default: 0)",
    )
    parser.add_argument(
        "-r",
        "--res",
        type=int,
        default=0,
        help="Which resolution level to load (default: 0)",
    )
    parser.add_argument(
        "--dask",
        action="store_true",
        help="Use to_dask() for lazy loading instead of as_array()",
    )
    parser.add_argument(
        "--xarray",
        action="store_true",
        help="Use to_xarray() for lazy loading instead of as_array()",
    )
    parser.add_argument(
        "--imread",
        action="store_true",
        help="Use imread() method for loading instead of as_array()"
        "(not recommended for large files)",
    )

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        raise SystemExit(1)

    if args.imread:
        data = imread(args.file_path, series=args.series, resolution=args.res)
        ndv.imshow(data)

    else:
        # Open the file with bffile
        with BioFile(args.file_path) as bf:
            if args.xarray:
                data = bf.to_xarray(series=args.series, resolution=args.res)
            elif args.dask:
                data = bf.to_dask(series=args.series, resolution=args.res)
            else:
                data = bf.as_array(series=args.series, resolution=args.res)
            ndv.imshow(data)


if __name__ == "__main__":
    main()
