"""Convenience function for reading image files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

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


@overload
def open_zarr(
    path: str | Path,
    *,
    series: Literal[None] = ...,
) -> zarr.Group: ...
@overload
def open_zarr(
    path: str | Path,
    *,
    series: int,
    resolution: int = 0,
) -> zarr.Array: ...
def open_zarr(
    path: str | Path, *, series: int | None = None, resolution: int = 0
) -> zarr.Array | zarr.Group:
    """Read image data from a Bio-Formats-supported file as a zarr array or group.

    Parameters
    ----------
    path : str or Path
        Path to the image file
    series : int or None, optional
        Series index to read. If `None` (default), returns an OME `zarr.Group` following
        the [bf2raw](https://ngff.openmicroscopy.org/0.5/index.html#bf2raw) transitional
        layout with all series. If an int, returns a `zarr.Array` for the specified
        series and resolution level.
    resolution : int, optional
        Resolution level (0 = full resolution), by default 0.
        Only used when series is an int.

    Returns
    -------
    zarr.Array or zarr.Group
        If series is None: zarr Group with full OME-ZARR hierarchy
        If series is int: zarr Array for the specified series

    Examples
    --------
    Open a single series as an array:

    >>> zarr_array = open_zarr("image.nd2", series=0)
    >>> data = zarr_array[0, 0, 0]

    Open all series as a group:

    >>> zarr_group = open_zarr("image.nd2")
    >>> # Access first series, full resolution
    >>> arr = zarr_group["0/0"]
    >>> data = arr[0, 0, 0]
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr must be installed to use open_zarr") from None

    with BioFile(path).ensure_open() as bf:
        if series is None:
            store = bf.as_zarr_group()
        else:
            store = bf.as_array(series=series, resolution=resolution).zarr_store()

    return zarr.open(store, mode="r")
