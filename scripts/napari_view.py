#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bffile[dask]",
#     "napari[pyqt6]",
# ]
#
# [tool.uv.sources]
# bffile = { path = "../" }
# ///
"""
View microscopy files using bffile and napari.

Usage:
    uv run scripts/napari_view.py <path_to_file> [options]

Examples:
    uv run scripts/napari_view.py image.nd2
    uv run scripts/napari_view.py image.czi --series 1 --res 0
    uv run scripts/napari_view.py large_file.nd2 --dask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import napari  # pyright: ignore[reportMissingImports]

from bffile import BioFile, imread
from bffile._utils import physical_pixel_sizes  # don't use, not public


def main() -> None:
    """Open a microscopy file with bffile and display it with napari.imshow()."""
    parser = argparse.ArgumentParser(
        description="View files using bffile and napari",
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
        default=None,
        type=int,
        help="Which resolution level to load (default: 0)",
    )
    parser.add_argument(
        "--dask",
        action="store_true",
        help="Use to_dask() for lazy loading instead of as_array()",
    )
    parser.add_argument(
        "--imread",
        action="store_true",
        help="Use imread() method for loading instead of as_array() or to_dask() "
        "(not recommended for large files)",
    )

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        raise SystemExit(1)

    if args.imread:
        data = imread(args.file_path, series=args.series, resolution=args.res or 0)
        napari.imshow(data, channel_axis=1)
        napari.run()

    else:
        # Open the file with bffile
        bf = BioFile(args.file_path).open()
        meta = bf.core_metadata(series=args.series)
        method = bf.to_dask if args.dask else bf.as_array
        scale = None
        if meta.resolution_count > 1:
            data = [
                method(series=args.series, resolution=res)
                for res in range(meta.resolution_count)
            ]
        else:
            data = method(series=args.series, resolution=args.res or 0)
            scale = [1] * (data.ndim - 1)
            pix = physical_pixel_sizes(bf.ome_metadata)
            if pix.z:
                scale[1] = pix.z
            if pix.y:
                scale[2] = pix.y
            if pix.x:
                scale[3] = pix.x

        napari.imshow(data, channel_axis=1, scale=scale)
        napari.run()


if __name__ == "__main__":
    main()
