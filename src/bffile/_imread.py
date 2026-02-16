"""Convenience function for reading image files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._biofile import BioFile

if TYPE_CHECKING:
    from pathlib import Path

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
        return np.asarray(arr)


def open_zarr_array(
    path: str | Path,
    *,
    series: int = 0,
    resolution: int = 0,
    rgb_as_channels: bool = False,
) -> zarr.Array:
    """Read image data from a Bio-Formats-supported file as a zarr array.

    By default, returns arrays with the same shape as `imread()`:
    - RGB images: `(T, C, Z, Y, X, rgb)`
    - Non-RGB images: `(T, C, Z, Y, X)`

    Set `rgb_as_channels=True` to interleave RGB samples as separate C channels
    (OME-Zarr convention), giving shape `(T, C*rgb, Z, Y, X)`.

    Parameters
    ----------
    path : str or Path
        Path to the image file
    series : int, optional
        Series index to read, by default 0.
    resolution : int, optional
        Resolution level (0 = full resolution), by default 0.
    rgb_as_channels : bool, optional
        If True, interleave RGB samples as separate C channels (OME-Zarr convention).
        If False, keep RGB as the last dimension (numpy/imread convention).
        Default is False.

    Returns
    -------
    zarr.Array
        Zarr array view of the image data

    Examples
    --------
    >>> import bffile
    >>> arr = bffile.open_zarr_array("image.jpg")
    >>> arr.shape  # RGB as last dimension (like imread)
    (1, 1, 1, 512, 512, 3)

    >>> arr = bffile.open_zarr_array("image.jpg", rgb_as_channels=True)
    >>> arr.shape  # RGB interleaved as channels (OME-Zarr style)
    (1, 3, 1, 512, 512)
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr must be installed to use open_zarr_array") from None

    with BioFile(path).ensure_open() as bf:
        store = bf.as_array(series=series, resolution=resolution).to_zarr_store(
            rgb_as_channels=rgb_as_channels
        )
    return zarr.open_array(store, mode="r")


def open_ome_zarr_group(
    path: str | Path, *, version: Literal["0.5"] = "0.5"
) -> zarr.Group:
    """Read image data from a Bio-Formats-supported file as a zarr array or group.

    Returns an OME `zarr.Group` following the
    [bf2raw](https://ngff.openmicroscopy.org/0.5/index.html#bf2raw) transitional
    layout with all series.

    Parameters
    ----------
    path : str or Path
        Path to the image file
    version : str, optional
        OME-ZARR version to use, by default "0.5". Only "0.5" is currently supported.
        This keyword is included for future compatibility when newer OME-ZARR
        versions are supported; pass it explicitly to avoid unexpected future changes.

    Examples
    --------
    >>> zarr_group = open_ome_zarr_group("image.nd2")
    >>> # Access first series, full resolution
    >>> arr = zarr_group["0/0"]
    >>> data = arr[0, 0, 0]
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr must be installed to use open_ome_zarr_group") from None

    with BioFile(path).ensure_open() as bf:
        store = bf.to_zarr_store()
    return zarr.open_group(store, mode="r")
