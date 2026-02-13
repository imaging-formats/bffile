"""Convenience function for reading image files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._biofile import BioFile

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import zarr


def imread(path: str | Path, *, series: int = 0, resolution: int = 0) -> np.ndarray:
    """Read image data from a Bio-Formats-supported file into a numpy array.

    Convenience function that opens a file, reads the specified series into
    memory, and returns it as a numpy array. For more control over reading
    (lazy loading, sub-regions, etc.), use BioFile directly.

    Parameters
    ----------
    path : str or Path
        Path to the image file
    series : int, optional
        Series index to read, by default 0
    resolution : int, optional
        Resolution level (0 = full resolution), by default 0

    Returns
    -------
    np.ndarray
        Image data with shape (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb)

    Examples
    --------
    >>> from bffile import imread
    >>> data = imread("image.nd2")
    >>> print(data.shape, data.dtype)
    (10, 2, 5, 512, 512) uint16

    Read a specific series:

    >>> data = imread("multi_series.czi", series=1)

    See Also
    --------
    BioFile : For lazy loading and more control over reading
    """
    with BioFile(path) as bf:
        arr = bf.as_array(series=series, resolution=resolution)
        return arr[:]


# TODO
# @overload
# def open_zarr(
#     path: str | Path,
#     *,
#     series: Literal[None] = ...,
#     resolution: Literal[None] = ...,
#     open_zarr: Literal[True],
# ) -> zarr.Group: ...
# @overload
# def open_zarr(
#     path: str | Path,
#     *,
#     series: int,
#     resolution: int = 0,
#     open_zarr: Literal[True],
# ) -> zarr.Array: ...
def open_zarr(
    path: str | Path, *, series: int | None, resolution: int = 0
) -> zarr.Array | zarr.Group:
    """Read image data from a Bio-Formats-supported file as a zarr array."""
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr must be installed to use open_zarr") from None

    if series is None:
        raise NotImplementedError(
            "open_zarr with series=None (all series in a group) is not yet implemented"
        )

    bf = BioFile(path).open()
    arr = bf.as_array(series=series, resolution=resolution)
    zarr_array = zarr.open_array(arr.zarr_store())
    bf.close()
    return zarr_array
