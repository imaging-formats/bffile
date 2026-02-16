"""Lightweight proxy representing a single series in a Bio-Formats file."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dask.array
    import numpy as np
    import xarray

    from bffile._biofile import BioFile
    from bffile._lazy_array import LazyBioArray
    from bffile._zarr._array_store import BFArrayStore


class Series:
    """Proxy for a single series within a [`BioFile`][bffile.BioFile].

    Provides convenient access to metadata and data for one series without
    needing to pass ``series=`` to every method call.

    Parameters
    ----------
    biofile : BioFile
        Open BioFile instance this series belongs to.
    index : int
        Zero-based series index.
    """

    __slots__ = ("_biofile", "_dtype", "_index", "_is_rgb", "_series_meta", "_shape")

    def __init__(self, biofile: BioFile, index: int) -> None:
        self._biofile = biofile
        self._index = index
        self._series_meta = biofile.core_metadata(index)

    @property
    def index(self) -> int:
        """Zero-based series index."""
        return self._index

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape as ``(T, C, Z, Y, X)`` or ``(T, C, Z, Y, X, rgb)``."""
        return self._series_meta.shape.as_array_shape

    @property
    def dtype(self) -> np.dtype:
        """Pixel data type at resolution 0."""
        return self._series_meta.dtype

    @property
    def is_rgb(self) -> bool:
        """Whether this series has multiple RGB components."""
        return self._series_meta.is_rgb

    @property
    def is_thumbnail(self) -> bool:
        """Whether this series is a thumbnail."""
        return self._series_meta.is_thumbnail_series

    @property
    def resolution_count(self) -> int:
        """Number of resolution levels in this series."""
        return self._series_meta.resolution_count

    def core_metadata(self, resolution: int = 0):
        """Return :class:`CoreMetadata` for this series.

        Parameters
        ----------
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        """
        return self._biofile.core_metadata(self._index, resolution)

    def as_array(self, resolution: int = 0) -> LazyBioArray:
        """Return a lazy array for this series.

        Parameters
        ----------
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        """
        return self._biofile.as_array(self._index, resolution)

    def to_dask(
        self,
        resolution: int = 0,
        *,
        chunks: str | tuple = "auto",
        tile_size: tuple[int, int] | str | None = None,
    ) -> dask.array.Array:
        """Create a dask array for this series.

        Parameters
        ----------
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        chunks : str or tuple, optional
            Chunk specification, by default "auto".
        tile_size : tuple[int, int] or "auto", optional
            Tile-based chunking for Y,X dimensions.
        """
        lazy = self.as_array(resolution=resolution)
        return lazy.to_dask(chunks=chunks, tile_size=tile_size)

    def to_xarray(self, resolution: int = 0) -> xarray.DataArray:
        """Create an xarray DataArray for this series.

        Parameters
        ----------
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        """
        lazy = self.as_array(resolution=resolution)
        return lazy.to_xarray()

    def to_zarr_store(
        self, resolution: int = 0, *, tile_size: tuple[int, int] | None = None
    ) -> BFArrayStore:
        """Create a Zarr store for this series.

        Parameters
        ----------
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        tile_size : tuple[int, int], optional
            If provided, Y and X are chunked into tiles of this size instead of
            full planes. Chunk shape becomes ``(1, 1, 1, tile_y, tile_x)``.
        """
        lazy = self.as_array(resolution=resolution)
        return lazy.to_zarr_store(tile_size=tile_size)

    def read_plane(
        self,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice | None = None,
        x: slice | None = None,
        resolution: int = 0,
        buffer: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read a single plane from this series.

        Parameters
        ----------
        t : int, optional
            Time index, by default 0.
        c : int, optional
            Channel index, by default 0.
        z : int, optional
            Z-slice index, by default 0.
        y : slice, optional
            Y-axis sub-region.
        x : slice, optional
            X-axis sub-region.
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.
        buffer : np.ndarray, optional
            Pre-allocated buffer for reuse.
        """
        return self._biofile.read_plane(
            t=t,
            c=c,
            z=z,
            y=y,
            x=x,
            series=self._index,
            resolution=resolution,
            buffer=buffer,
        )

    def __repr__(self) -> str:
        return f"Series(index={self._index}, shape={self.shape}, dtype={self.dtype})"

    def used_files(self, *, metadata_only: bool = False) -> list[str]:
        """Return list of files needed to open this series.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only return files that do not contain pixel data (e.g., metadata,
            companion files, etc...), by default `False`.
        """
        reader = self._biofile._ensure_java_reader()
        reader.setSeries(self._index)
        return [str(f) for f in reader.getSeriesUsedFiles(metadata_only) or ()]
