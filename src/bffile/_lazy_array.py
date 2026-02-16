"""Lazy numpy-compatible array for on-demand Bio-Formats reading."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import dask
import numpy as np

from bffile._utils import get_dask_tile_chunks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import dask.array
    import xarray as xr

    from bffile._biofile import BioFile
    from bffile._zarr import BFArrayStore


BoundsTCZYXS: TypeAlias = tuple[slice, slice, slice, slice, slice, slice]
SqueezedTCZYXS: TypeAlias = tuple[bool, bool, bool, bool, bool, bool]
ShapeTCZYXS: TypeAlias = tuple[int, int, int, int, int, int]


class LazyBioArray:
    """Pythonic lazy array interface for a single Bio-Formats Series/Resolution.

    This object provides a numpy-compatible API for on-demand access to a
    specific series and resolution level in a Bio-Formats file. In the
    Bio-Formats Java API, each file can contain multiple series (e.g., wells
    in a plate, fields of view, or tiled regions), and each series can have
    multiple resolution levels (pyramid layers). LazyBioArray represents one
    of these series/resolution combinations as a numpy-style array.

    The array is always 5-dimensional with shape (T, C, Z, Y, X), though some
    dimensions may be singletons (size 1) depending on the image acquisition.
    For RGB/RGBA images, a 6th dimension is added: (T, C, Z, Y, X, rgb).

    **Lazy slicing behavior:** Indexing operations (`arr[...]`) return lazy views
    without reading data. Use `np.asarray()` or numpy operations to materialize.
    This enables composition: `arr[0][1][2]` creates nested views, only reading
    data when explicitly requested.

    Supports integer and slice indexing along with the `__array__()` protocol
    for seamless numpy integration.

    Parameters
    ----------
    biofile : BioFile
        BioFile instance to read from. Must remain open during use.

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape in (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb) format
    dtype : np.dtype
        Data type of array elements
    ndim : int
        Number of dimensions (5 for grayscale, 6 for RGB)
    size : int
        Number of elements in the array
    nbytes : int
        Total bytes consumed by the array elements

    Examples
    --------
    >>> with BioFile("image.nd2") as bf:
    ...     arr = bf.as_array()  # No data read yet
    ...     view = arr[0, 0, 2]  # Returns LazyBioArray view (no I/O)
    ...     plane = np.asarray(view)  # Now reads single plane from disk
    ...     roi = arr[:, :, :, 100:200, 50:150]  # Lazy view of sub-region
    ...     full_data = np.array(arr)  # Materialize all data
    ...     max_z = np.max(arr, axis=2)  # Works with numpy functions

    Composition example:

    >>> with BioFile("image.nd2") as bf:
    ...     arr = bf.as_array()
    ...     view1 = arr[0:10]  # LazyBioArray (no I/O)
    ...     view2 = view1[2:5]  # LazyBioArray (still no I/O)
    ...     data = np.asarray(view2)  # Read frames 2-4 from disk

    Notes
    -----
    - BioFile must remain open while using this array
    - Step indexing (`arr[::2]`), fancy indexing, and boolean masks not supported
    - Not thread-safe: create separate BioFile instances per thread
    """

    __slots__ = (
        "_biofile",
        "_bounds_tczyxs",
        "_dtype",
        "_full_shape_tczyxs",
        "_meta",
        "_resolution",
        "_series",
        "_shape",
        "_squeezed_tczyxs",
    )

    def __init__(self, biofile: BioFile, series: int, resolution: int = 0) -> None:
        """
        Initialize lazy array wrapper.

        Parameters
        ----------
        biofile : BioFile
            Open BioFile instance to read from
        series : int
            Series index this array represents
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0
        """
        self._biofile = biofile
        self._series = series
        self._resolution = resolution

        # Get metadata directly from the 2D list (stateless!)
        # This avoids hidden dependency on biofile's current state
        self._meta = meta = biofile.core_metadata(series, resolution)
        self._dtype = meta.dtype
        self._full_shape_tczyxs = full = cast("ShapeTCZYXS", tuple(self._meta.shape))

        # View state tracking (for lazy slicing)
        # Initialize to full range (root array shows entire dataset)
        self._bounds_tczyxs = cast("BoundsTCZYXS", tuple(slice(0, x) for x in full))
        self._squeezed_tczyxs = (False, False, False, False, False, meta.shape.rgb <= 1)
        self._shape = self._effective_shape()

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb) format."""
        return self._shape

    @property
    def size(self) -> int:
        """Number of elements in the array"""
        return math.prod(self._shape)

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the array elements."""
        return self.size * self.dtype.itemsize

    @property
    def dtype(self) -> np.dtype:
        """Data type of array elements."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def is_rgb(self) -> bool:
        """True if image has RGB/RGBA components (ndim == 6)."""
        return self._meta.is_rgb and not self._squeezed_tczyxs[5]

    @property
    def dims(self) -> tuple[str, ...]:
        """Return used dimension names (matches shape)."""
        return tuple(
            dim
            for dim, squeezed in zip("TCZYXS", self._squeezed_tczyxs, strict=False)
            if not squeezed
        )

    @property
    def coords(self) -> Mapping[str, Any]:
        """Mapping of dimension names to coordinate values for this view.

        Squeezed dimensions are returned as scalars, non-squeezed dimensions are
        returned as ranges or sequences of values.

        This mimics the .coords attribute of xarray.DataArray for compatibility
        with xarray's indexing
        """
        # build coords from bounds
        coords: dict[str, Sequence[Any]] = {
            dim: range(*bound.indices(size))
            for dim, bound, size in zip(
                "TCZYXS",
                self._bounds_tczyxs,
                self._full_shape_tczyxs,
                strict=False,
            )
        }

        if self._meta.shape.rgb > 1:
            rgba = ["R", "G", "B", "A"]
            coords["S"] = [rgba[i] if i < len(rgba) else f"S{i}" for i in coords["S"]]
        else:
            coords.pop("S", None)

        # Apply scene pixels metadata if possible
        try:
            pix = self._biofile.ome_metadata.images[self._series].pixels
        except (IndexError, AttributeError):
            pass
        else:
            # convert channel range to actual names
            if pix.channels:
                coords["C"] = [pix.channels[ci].name or f"C{ci}" for ci in coords["C"]]

            planes = pix.planes
            t_map = {p.the_t: p.delta_t for p in planes if p.the_t in coords["T"]}
            if all(delta is not None for delta in t_map.values()):
                # we have actual timestamps
                coords["T"] = [t_map.get(t, 0) for t in coords["T"]]
            elif pix.time_increment is not None:
                # otherwise fall back to global time increment if available
                coords["T"] = [t * float(pix.time_increment) for t in coords["T"]]

            # Spatial coordinates - use physical sizes if available
            if (pz := pix.physical_size_z) is not None:
                coords["Z"] = np.asarray(coords["Z"]) * float(pz)  # type: ignore
            if (py := pix.physical_size_y) is not None:
                coords["Y"] = np.asarray(coords["Y"]) * float(py)  # type: ignore
            if (px := pix.physical_size_x) is not None:
                coords["X"] = np.asarray(coords["X"]) * float(px)  # type: ignore

        # now squeeze any dimensions that are marked as squeezed into scalars
        for dim, squeezed in zip("TCZYXS", self._squeezed_tczyxs, strict=False):
            if squeezed and dim in coords:
                coords[dim] = coords[dim][0]

        return coords

    def to_dask(
        self,
        *,
        chunks: str | tuple = "auto",
        tile_size: tuple[int, int] | str | None = None,
    ) -> dask.array.Array:
        """Create dask array for lazy computation on Bio-Formats data.

        Returns a dask array in TCZYX[r] order that wraps this lazy array.
        Uses single-threaded scheduler for Bio-Formats thread safety.

        Parameters
        ----------
        chunks : str or tuple, default "auto"
            Chunk specification. Examples:
            - "auto": Let dask decide (default)
            - (1, 1, 1, -1, -1): Full Y,X planes per T,C,Z
            - (1, 1, 1, 512, 512): 512x512 tiles
            Mutually exclusive with tile_size.
        tile_size : tuple[int, int] or "auto", optional
            Tile-based chunking for Y,X dimensions (T,C,Z get chunks of 1).
            - (512, 512): Use 512x512 tiles
            - "auto": Query Bio-Formats optimal tile size
            Mutually exclusive with chunks.
        """
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Dask is required for to_dask(). "
                "Please install with `pip install bffile[dask]`"
            ) from e

        # Validate mutually exclusive parameters
        if tile_size is not None and chunks != "auto":
            raise ValueError(
                "chunks and tile_size are mutually exclusive. "
                "When using tile_size, leave chunks as 'auto' (default)."
            )

        # Compute chunks from tile_size if provided
        if tile_size is not None:
            # Validate tile_size format
            if tile_size == "auto":
                # Query Bio-Formats for optimal tile size
                rdr = self._biofile._ensure_java_reader()
                rdr.setSeries(self._series)
                rdr.setResolution(self._resolution)
                tile_size = (rdr.getOptimalTileHeight(), rdr.getOptimalTileWidth())
            elif not (
                isinstance(tile_size, tuple)
                and len(tile_size) == 2
                and all(isinstance(x, int) for x in tile_size)
            ):
                raise ValueError(
                    f"tile_size must be a tuple of two integers or 'auto', "
                    f"got {tile_size}"
                )

            # Compute chunks based on tile size
            # Use the view's actual shape (from bounds), not the full original shape
            nt, nc, nz, ny, nx, nrgb = self._unsqueezed_view_shape()
            chunks = get_dask_tile_chunks(nt, nc, nz, ny, nx, tile_size)
            if nrgb > 1:
                chunks = (*chunks, nrgb)  # type: ignore[assignment]

        return da.from_array(self, chunks=chunks)  # type: ignore

    def to_xarray(self) -> xr.DataArray:
        """Return xarray.DataArray for specified series and resolution.

        The returned DataArray has `.dims` and `.coords` attributes populated according
        to the metadata. Dimension and coord names will be one of: `TCZYXS`, where `S`
        represents the RGB/RGBA channels if present.  The parsed `ome_types.OME` object
        is also available in the `.attrs['ome_metadata']` attribute of the DataArray.
        """
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError(
                "xarray is required for to_xarray(). "
                "Install with `pip install bffile[xarray]`"
            ) from e

        return xr.DataArray(
            self,
            dims=self.dims,
            coords=self.coords,
            attrs={"ome_metadata": self._biofile.ome_metadata},
        )

    def to_zarr_store(
        self,
        *,
        tile_size: tuple[int, int] | None = None,
        rgb_as_channels: bool = False,
        squeeze_singletons: bool = False,
    ) -> BFArrayStore:
        """Create a read-only zarr v3 store backed by this array.

        Each zarr chunk maps to a single ``read_plane()`` call. Requires
        the ``zarr`` extra (``pip install bffile[zarr]``).

        Parameters
        ----------
        tile_size : tuple[int, int], optional
            If provided, Y and X are chunked into tiles of this size.
            Default is full-plane chunks ``(1, 1, 1, Y, X)``.
        rgb_as_channels : bool, optional
            If True, interleave RGB samples as separate C channels (OME-Zarr
            convention). If False, keep RGB as the last dimension (numpy/imread
            convention). Default is False.
        squeeze_singletons : bool, optional
            If True, omit dimensions with size 1 from metadata (except Y/X).
            Default is False (always reports 5D or 6D arrays).

        Returns
        -------
        BFArrayStore
            A zarr v3 Store suitable for ``zarr.open_array(store, mode="r")``.
        """
        from bffile._zarr import BFArrayStore

        return BFArrayStore(
            self._biofile,
            self._series,
            self._resolution,
            tile_size=tile_size,
            rgb_as_channels=rgb_as_channels,
            squeeze_singletons=squeeze_singletons,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LazyBioArray(shape={self.shape}, dtype={self.dtype}, "
            f"file='{self._biofile.filename}')"
        )

    def __getitem__(self, key: Any) -> LazyBioArray:
        """Index the array with numpy-style syntax, returning a lazy view.

        Supports integer and slice indexing. Returns a view without reading data -
        use np.asarray() or __array__() to materialize.

        Parameters
        ----------
        key : int, slice, tuple, or Ellipsis
            Index specification

        Returns
        -------
        LazyBioArray
            A lazy view of the requested data

        Raises
        ------
        NotImplementedError
            If fancy indexing, boolean indexing, or step != 1 is used
        IndexError
            If indices are out of bounds
        """
        # Map user's effective-space index to original TCZYXS coordinates
        key = self._normalize_key(key)
        new_bounds, new_squeezed = self._map_user_index_to_tczyxs(key)

        return LazyBioArray._create_view(
            parent=self,
            bounds_tczyxs=new_bounds,
            squeezed_tczyxs=new_squeezed,
        )

    # ============================ Numpy array protocol ===========================

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """numpy array protocol - materializes data from disk.

        This enables `np.array(lazy_arr)` and other numpy operations.

        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type
        copy : bool, optional
            Whether to force a copy (NumPy 2.0+ compatibility)
        """
        output = np.empty(self._shape, dtype=self._dtype)
        selection = self._bounds_to_selection()
        self._fill_output(output, selection, self._squeezed_tczyxs)
        if dtype is not None and output.dtype != dtype:
            output = output.astype(dtype, copy=False)

        # data is always fresh from disk so no copy needed
        # but honor explicit copy=True request
        if copy:
            output = output.copy()

        return output

    def __array_function__(
        self, func: Callable, types: list[type], args: tuple, kwargs: dict
    ) -> Any:
        # just dispatch to numpy for now - this allows xarray to be lazy
        # but we could implement some functions natively here in the future if desired
        def convert_arg(a: Any) -> Any:
            """Recursively convert LazyBioArray instances to numpy arrays."""
            if isinstance(a, type(self)):
                return np.asarray(a)
            if isinstance(a, (list, tuple)):
                return type(a)(convert_arg(item) for item in a)
            return a

        args = tuple(convert_arg(a) for a in args)
        kwargs = {k: convert_arg(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    # this is a hack to allow this object to work with dask da.from_array
    # dask calls it during `compute_meta` ...
    # this should NOT be used for any other purpose, it does NOT do what it claims to do
    # if we directly used dask.map_blocks again we could lose this...
    def astype(self, dtype: np.dtype) -> Any:
        return self

    # ============================= Private methods =============================

    @classmethod
    def _create_view(
        cls,
        parent: LazyBioArray,
        bounds_tczyxs: BoundsTCZYXS,
        squeezed_tczyxs: SqueezedTCZYXS,
    ) -> LazyBioArray:
        """Create a view of a parent array without reading data."""
        view = cls.__new__(cls)
        view._meta = meta = parent._meta
        view._biofile = parent._biofile
        view._series = parent._series
        view._resolution = parent._resolution
        view._full_shape_tczyxs = cast("ShapeTCZYXS", tuple(meta.shape))
        view._dtype = meta.dtype
        view._bounds_tczyxs = bounds_tczyxs
        view._squeezed_tczyxs = squeezed_tczyxs
        view._shape = view._effective_shape()
        return view

    def _unsqueezed_view_shape(self) -> tuple[int, int, int, int, int, int]:
        """Compute full 6D TCZYXS shape from bounds (includes squeezed dims)."""
        return tuple(  # type: ignore[return-value]  (it is always 6D)
            bound.indices(size)[1] - bound.indices(size)[0]
            for size, bound in zip(
                self._full_shape_tczyxs, self._bounds_tczyxs, strict=True
            )
        )

    def _effective_shape(self) -> tuple[int, ...]:
        """Compute visible shape from bounds, excluding squeezed dimensions."""
        view_shape = self._unsqueezed_view_shape()
        return tuple(
            size
            for size, squeezed in zip(view_shape, self._squeezed_tczyxs, strict=True)
            if not squeezed
        )

    def _map_user_index_to_tczyxs(
        self, key: tuple[slice | int, ...]
    ) -> tuple[BoundsTCZYXS, SqueezedTCZYXS]:
        new_bounds = list(self._bounds_tczyxs)
        new_squeezed = list(self._squeezed_tczyxs)
        key_iter = iter(key)

        for dim_idx, (bound, size, squeezed_now) in enumerate(
            zip(
                self._bounds_tczyxs,
                self._full_shape_tczyxs,
                self._squeezed_tczyxs,
                strict=True,
            )
        ):
            if squeezed_now:
                continue
            new_bound, squeezed = _compose_index(next(key_iter), bound, size)
            new_bounds[dim_idx] = new_bound
            new_squeezed[dim_idx] = squeezed

        return tuple(new_bounds), tuple(new_squeezed)  # type: ignore[return-value]

    def _bounds_to_selection(self) -> tuple[range, range, range, slice, slice, slice]:
        """Convert stored bounds to selection format for _fill_output.

        Returns (t_range, c_range, z_range, y_slice, x_slice, s_slice).
        """
        t_slice, c_slice, z_slice, y_slice, x_slice, s_slice = self._bounds_tczyxs
        t_size, c_size, z_size, *_ = self._full_shape_tczyxs

        # Convert slices to ranges for TCZ iteration
        t_start, t_stop, _ = t_slice.indices(t_size)
        c_start, c_stop, _ = c_slice.indices(c_size)
        z_start, z_stop, _ = z_slice.indices(z_size)

        t_range = range(t_start, t_stop)
        c_range = range(c_start, c_stop)
        z_range = range(z_start, z_stop)

        # Y, X, and S stay as slices
        return t_range, c_range, z_range, y_slice, x_slice, s_slice

    def _normalize_key(self, key: Any) -> tuple[slice | int, ...]:
        """Normalize indexing key to tuple of slices/ints.

        Handles scalars, tuples, and ellipsis expansion.
        """
        # Convert scalar to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Check for unsupported indexing types FIRST (before ellipsis check)
        for k in key:
            if isinstance(k, list):
                msg = "fancy indexing with lists is not supported"
                raise NotImplementedError(msg)
            if isinstance(k, np.ndarray):
                msg = "fancy indexing with arrays is not supported"
                raise NotImplementedError(msg)
            if isinstance(k, slice):
                if k.step is not None and k.step != 1:
                    msg = f"step != 1 is not supported (got step={k.step})"
                    raise NotImplementedError(msg)

        # Handle ellipsis
        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            # Count non-ellipsis dimensions
            n_specified = len(key) - 1  # -1 for the ellipsis itself
            n_missing = self.ndim - n_specified
            # Replace ellipsis with appropriate number of full slices
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_missing
                + key[ellipsis_idx + 1 :]
            )

        # Pad with full slices if needed
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        # Validate length
        if len(key) > self.ndim:
            msg = (
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
            raise IndexError(msg)

        return key

    def _fill_output(
        self,
        output: np.ndarray,
        selection: tuple[range, range, range, slice, slice, slice],
        squeezed: SqueezedTCZYXS | list[bool],
    ) -> None:
        """Fill output array by reading planes from Bio-Formats.

        Parameters
        ----------
        output : np.ndarray
            Pre-allocated output array to fill
        selection : tuple
            (t_range, c_range, z_range, y_slice, x_slice, s_slice)
        squeezed : list[bool]
            Which TCZYXS dimensions are squeezed (True = squeezed)
        """
        t_range, c_range, z_range, y_slice, x_slice, s_slice = selection

        # Check if any dimension is empty (no data to read)
        if len(t_range) == 0 or len(c_range) == 0 or len(z_range) == 0:
            return
        y_start, y_stop, _ = y_slice.indices(self._meta.shape.y)
        x_start, x_stop, _ = x_slice.indices(self._meta.shape.x)
        if y_stop <= y_start or x_stop <= x_start:
            return

        bf = self._biofile
        # Acquire lock once for entire batch read
        with bf._lock:
            # Set series and resolution once at start (not on every iteration)
            reader = bf._ensure_java_reader()
            reader.setSeries(self._series)
            reader.setResolution(self._resolution)
            meta = self._meta
            read_plane = bf._read_plane

            # Pre-compute specialized writer to avoid tuple building on each write
            write_plane = _make_plane_writer(output, *squeezed[:3])

            s_index: int | slice | None = None
            if self._meta.shape.rgb > 1:
                if squeezed[5]:
                    s_start, _s_stop, _ = s_slice.indices(self._meta.shape.rgb)
                    s_index = s_start
                else:
                    s_index = s_slice

            for (ti, t), (ci, c), (zi, z) in product(
                enumerate(t_range),
                enumerate(c_range),
                enumerate(z_range),
            ):
                plane = read_plane(reader, meta, t, c, z, y_slice, x_slice)
                if s_index is not None:
                    plane = plane[..., s_index]
                write_plane(ti, ci, zi, plane)


def _make_plane_writer(
    output: np.ndarray, drop_t: bool, drop_c: bool, drop_z: bool
) -> Callable[[int, int, int, np.ndarray], None]:
    match (drop_t, drop_c, drop_z):
        case (False, False, False):
            return lambda t, c, z, plane: output.__setitem__((t, c, z), plane)
        case (False, False, True):
            return lambda t, c, z, plane: output.__setitem__((t, c), plane)
        case (False, True, False):
            return lambda t, c, z, plane: output.__setitem__((t, z), plane)
        case (False, True, True):
            return lambda t, c, z, plane: output.__setitem__((t,), plane)
        case (True, False, False):
            return lambda t, c, z, plane: output.__setitem__((c, z), plane)
        case (True, False, True):
            return lambda t, c, z, plane: output.__setitem__((c,), plane)
        case (True, True, False):
            return lambda t, c, z, plane: output.__setitem__((z,), plane)
        case (True, True, True):
            return lambda t, c, z, plane: output.__setitem__(slice(None), plane)


def _compose_index(
    user_idx: int | slice, parent_bound: slice, full_size: int
) -> tuple[slice, bool]:
    """Compose user index with parent bound, return (new_bound, squeezed).

    Examples
    --------
    >>> _compose_index(slice(2, 5), slice(10, 20), 100)
    (slice(12, 15, None), False)

    >>> _compose_index(3, slice(10, 20), 100)
    (slice(13, 14, None), True)
    """
    p_start, p_stop, _ = parent_bound.indices(full_size)
    p_size = p_stop - p_start

    if isinstance(user_idx, int):
        idx = user_idx if user_idx >= 0 else p_size + user_idx
        if not 0 <= idx < p_size:
            msg = f"index {user_idx} is out of bounds for size {p_size}"
            raise IndexError(msg)
        abs_idx = p_start + idx
        return slice(abs_idx, abs_idx + 1), True

    start, stop, step = user_idx.indices(p_size)
    if step != 1:
        msg = f"step != 1 is not supported (got step={step})"
        raise NotImplementedError(msg)
    return slice(p_start + start, p_start + stop), False
