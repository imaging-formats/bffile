"""Convenience function for reading image files."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

from ._biofile import BioFile


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
