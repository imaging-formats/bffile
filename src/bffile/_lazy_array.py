"""Lazy numpy-compatible array for on-demand Bio-Formats reading."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from bffile._biofile import BioFile
    from bffile._zarr import BFArrayStore


BoundsTCZYX: TypeAlias = tuple[slice, slice, slice, slice, slice]
SqueezedTCZYX: TypeAlias = tuple[bool, bool, bool, bool, bool]
RGBIndex: TypeAlias = slice | int | None


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
        "_bounds_tczyx",
        "_dtype",
        "_meta",
        "_resolution",
        "_rgb_index",
        "_series",
        "_shape",
        "_squeezed_tczyx",
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

        # Follow same logic as to_numpy(): only include RGB dimension if > 1
        self._shape = meta.shape.as_array_shape
        self._dtype = meta.dtype

        # View state tracking (for lazy slicing)
        # Initialize to full range (root array shows entire dataset)
        self._bounds_tczyx = (
            slice(0, meta.shape.t),
            slice(0, meta.shape.c),
            slice(0, meta.shape.z),
            slice(0, meta.shape.y),
            slice(0, meta.shape.x),
        )
        self._squeezed_tczyx = (False, False, False, False, False)
        self._rgb_index = slice(None) if meta.shape.rgb > 1 else None

    def dimension_names(self) -> tuple[str, ...]:
        """Return dimension names (matches shape)."""
        names = [
            dim
            for dim, squeezed in zip("TCZYX", self._squeezed_tczyx, strict=False)
            if not squeezed
        ]
        if self._rgb_index is not None:
            names.append("S")
        return tuple(names)

    @classmethod
    def _create_view(
        cls,
        parent: LazyBioArray,
        bounds_tczyx: BoundsTCZYX,
        squeezed_tczyx: SqueezedTCZYX,
        rgb_index: RGBIndex,
    ) -> LazyBioArray:
        """Create a view of a parent array without reading data."""
        view = cls.__new__(cls)
        view._biofile = parent._biofile
        view._series = parent._series
        view._resolution = parent._resolution
        view._meta = parent._meta
        view._dtype = parent._dtype
        view._bounds_tczyx = bounds_tczyx
        view._squeezed_tczyx = squeezed_tczyx
        view._rgb_index = rgb_index
        # Compute effective shape from bounds
        view._shape = view._compute_effective_shape()
        return view

    def _full_sizes_tczyx(self) -> tuple[int, int, int, int, int]:
        """Return full TCZYX sizes from metadata."""
        return (
            self._meta.shape.t,
            self._meta.shape.c,
            self._meta.shape.z,
            self._meta.shape.y,
            self._meta.shape.x,
        )

    def _compute_effective_shape(self) -> tuple[int, ...]:
        """Compute visible shape from bounds, excluding squeezed dimensions."""
        shape = []
        full_sizes = self._full_sizes_tczyx()

        # Add non-squeezed TCZYX dimensions
        for dim_idx in range(5):
            if not self._squeezed_tczyx[dim_idx]:
                bound = self._bounds_tczyx[dim_idx]
                full_size = full_sizes[dim_idx]
                start, stop, _ = bound.indices(full_size)
                shape.append(stop - start)

        # Handle RGB dimension
        if isinstance(self._rgb_index, slice):
            rgb_start, rgb_stop, _ = self._rgb_index.indices(self._meta.shape.rgb)
            shape.append(rgb_stop - rgb_start)

        return tuple(shape)

    def _split_key(
        self, key: tuple[slice | int, ...]
    ) -> tuple[tuple[slice | int, ...], slice | int | None]:
        """Split normalized key into TCZYX and RGB parts."""
        # Determine how many effective dimensions we have
        n_effective = sum(not s for s in self._squeezed_tczyx)
        has_rgb = self._rgb_index is not None and not isinstance(self._rgb_index, int)

        if has_rgb:
            # Last dimension is RGB if present
            if len(key) > n_effective:
                return key[:n_effective], key[n_effective]
            return key, slice(None)
        else:
            # No RGB dimension
            return key, None

    def _compose_index(
        self, user_idx: int | slice, parent_bound: slice, full_size: int
    ) -> tuple[slice, bool]:
        """Compose user index with parent bound, return (new_bound, squeezed)."""
        # Get parent's effective range
        p_start, p_stop, _ = parent_bound.indices(full_size)
        p_size = p_stop - p_start

        if isinstance(user_idx, int):
            # Resolve negative indices relative to effective size
            idx = user_idx if user_idx >= 0 else p_size + user_idx
            if idx < 0 or idx >= p_size:
                msg = f"index {user_idx} is out of bounds for size {p_size}"
                raise IndexError(msg)
            # Map to absolute position
            abs_idx = p_start + idx
            return slice(abs_idx, abs_idx + 1), True

        else:  # slice
            # Get effective range within parent
            start, stop, step = user_idx.indices(p_size)
            if step != 1:
                msg = f"step != 1 is not supported (got step={step})"
                raise NotImplementedError(msg)
            # Map to absolute positions
            abs_start = p_start + start
            abs_stop = p_start + stop
            return slice(abs_start, abs_stop), False

    def _map_user_index_to_tczyx(
        self, key: tuple[slice | int, ...]
    ) -> tuple[BoundsTCZYX, SqueezedTCZYX, RGBIndex]:
        """Map user's effective-space index to original TCZYX coordinates."""
        tczyx_key, rgb_key = self._split_key(key)

        # Build mapping from effective dimension index to TCZYX dimension
        effective_to_tczyx = []
        for dim_idx in range(5):
            if not self._squeezed_tczyx[dim_idx]:
                effective_to_tczyx.append(dim_idx)

        # Compose each user index with parent bounds
        new_bounds: list[slice] = list(self._bounds_tczyx)
        new_squeezed: list[bool] = list(self._squeezed_tczyx)
        full_sizes = self._full_sizes_tczyx()

        for eff_idx, user_idx in enumerate(tczyx_key):
            tczyx_idx = effective_to_tczyx[eff_idx]
            parent_bound = self._bounds_tczyx[tczyx_idx]
            full_size = full_sizes[tczyx_idx]

            new_bound, squeezed = self._compose_index(user_idx, parent_bound, full_size)
            new_bounds[tczyx_idx] = new_bound
            # Only mark as squeezed if parent wasn't already squeezed
            # AND this operation squeezes it
            new_squeezed[tczyx_idx] = self._squeezed_tczyx[tczyx_idx] or squeezed

        # Handle RGB
        new_rgb = self._rgb_index
        if rgb_key is not None and self._rgb_index is not None:
            rgb_size = self._meta.shape.rgb
            if isinstance(self._rgb_index, slice):
                p_start, p_stop, _ = self._rgb_index.indices(rgb_size)
                p_size = p_stop - p_start

                if isinstance(rgb_key, int):
                    idx = rgb_key if rgb_key >= 0 else p_size + rgb_key
                    if idx < 0 or idx >= p_size:
                        msg = f"RGB index {rgb_key} is out of bounds"
                        raise IndexError(msg)
                    new_rgb = p_start + idx
                else:
                    start, stop, step = rgb_key.indices(p_size)
                    if step != 1:
                        msg = f"step != 1 is not supported (got step={step})"
                        raise NotImplementedError(msg)
                    new_rgb = slice(p_start + start, p_start + stop)
            elif isinstance(self._rgb_index, int):
                msg = "RGB dimension is already squeezed"
                raise IndexError(msg)

        return (
            tuple(new_bounds),  # type: ignore[return-value]
            tuple(new_squeezed),  # type: ignore[return-value]
            new_rgb,
        )

    def _bounds_to_selection(self) -> tuple[range, range, range, slice, slice]:
        """Convert stored bounds to selection format for _fill_output."""
        t_slice, c_slice, z_slice, y_slice, x_slice = self._bounds_tczyx
        t_size, c_size, z_size, _y_size, _x_size = self._full_sizes_tczyx()

        # Convert slices to ranges for TCZ iteration
        t_start, t_stop, _ = t_slice.indices(t_size)
        c_start, c_stop, _ = c_slice.indices(c_size)
        z_start, z_stop, _ = z_slice.indices(z_size)

        t_range = range(t_start, t_stop)
        c_range = range(c_start, c_stop)
        z_range = range(z_start, z_stop)

        # Y and X stay as slices
        return t_range, c_range, z_range, y_slice, x_slice

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
        return self.ndim == 6

    def zarr_store(
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
        BioFormatsStore
            A zarr v3 Store suitable for ``zarr.open(store, mode="r")``.
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
        # Convert stored bounds to selection format
        selection = self._bounds_to_selection()

        # Allocate output with effective shape
        output = np.empty(self._shape, dtype=self._dtype)
        # Fill using existing optimized _fill_output (preserves locking!)
        self._fill_output(output, selection, self._squeezed_tczyx)

        # Handle dtype conversion if needed
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
        # Normalize key to tuple
        key = self._normalize_key(key)

        # Map user's effective-space index to original TCZYX coordinates
        new_bounds, new_squeezed, new_rgb = self._map_user_index_to_tczyx(key)

        # Create new view (no data read!)
        return LazyBioArray._create_view(
            parent=self,
            bounds_tczyx=new_bounds,
            squeezed_tczyx=new_squeezed,
            rgb_index=new_rgb,
        )

    def _normalize_key(self, key: Any) -> tuple[slice | int, ...]:
        """Normalize indexing key to tuple of slices/ints.

        Handles scalars, tuples, ellipsis expansion, and RGB dimension.
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
        selection: tuple[range, range, range, slice, slice],
        squeezed: SqueezedTCZYX | list[bool],
    ) -> None:
        """Fill output array by reading planes from Bio-Formats.

        Parameters
        ----------
        output : np.ndarray
            Pre-allocated output array to fill
        selection : tuple
            (t_range, c_range, z_range, y_slice, x_slice)
        squeezed : list[bool]
            Which TCZYX dimensions are squeezed (True = squeezed)
        """
        t_range, c_range, z_range, y_slice, x_slice = selection

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

            # Get metadata once (avoid repeated lookups in hot loop)
            meta = bf.core_metadata(self._series, self._resolution)
            read_plane = bf._read_plane

            # Pre-compute specialized writer to avoid tuple building on each write
            write_plane = _make_plane_writer(output, *squeezed[:3])

            rgb_index = self._rgb_index
            for (ti, t), (ci, c), (zi, z) in product(
                enumerate(t_range), enumerate(c_range), enumerate(z_range)
            ):
                plane = read_plane(reader, meta, t, c, z, y_slice, x_slice)
                if rgb_index is not None:
                    plane = plane[..., rgb_index]
                write_plane(ti, ci, zi, plane)

    def _build_coords(self) -> dict[str, Any]:
        """Build coordinates (suitable for xarray).

        Squeezed dimensions are returned as scalars, non-squeezed dimensions are
        returned as ranges or sequences of values.
        """
        # build coords from bounds
        coords: dict[str, Sequence[Any]] = {
            dim: range(*bound.indices(size))
            for dim, bound, size in zip(
                "TCZYX",
                self._bounds_tczyx,
                self._meta.shape.as_array_shape,
                strict=False,
            )
        }

        if self._rgb_index is not None:
            RGBA = ["R", "G", "B", "A"][: self._meta.shape.rgb]
            coords["S"] = RGBA[self._rgb_index]

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
        for dim, squeezed in zip("TCZYX", self._squeezed_tczyx, strict=False):
            if squeezed and dim in coords:
                coords[dim] = coords[dim][0]

        return coords


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
