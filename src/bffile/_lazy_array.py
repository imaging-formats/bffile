"""Lazy numpy-compatible array for on-demand Bio-Formats reading."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bffile._biofile import BioFile
    from bffile._zarr import BFArrayStore


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

    @classmethod
    def _create_view(
        cls,
        parent: LazyBioArray,
        bounds_tczyx: tuple[slice, slice, slice, slice, slice],
        squeezed_tczyx: tuple[bool, bool, bool, bool, bool],
        rgb_index: slice | int | None,
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

    def _compute_effective_shape(self) -> tuple[int, ...]:
        """Compute visible shape from bounds, excluding squeezed dimensions."""
        shape = []

        # Add non-squeezed TCZYX dimensions
        for dim_idx in range(5):
            if not self._squeezed_tczyx[dim_idx]:
                bound = self._bounds_tczyx[dim_idx]
                # Get the size from the full metadata shape
                full_size = (
                    self._meta.shape.t,
                    self._meta.shape.c,
                    self._meta.shape.z,
                    self._meta.shape.y,
                    self._meta.shape.x,
                )[dim_idx]
                start, stop, _ = bound.indices(full_size)
                shape.append(stop - start)

        # Handle RGB dimension
        if self._rgb_index is not None and not isinstance(self._rgb_index, int):
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
    ) -> tuple[
        tuple[slice, slice, slice, slice, slice],
        tuple[bool, bool, bool, bool, bool],
        slice | int | None,
    ]:
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
        full_sizes = (
            self._meta.shape.t,
            self._meta.shape.c,
            self._meta.shape.z,
            self._meta.shape.y,
            self._meta.shape.x,
        )

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

    def _bounds_to_selection(
        self,
    ) -> tuple[range, range, range, slice, slice]:
        """Convert stored bounds to selection format for _fill_output."""
        t_slice, c_slice, z_slice, y_slice, x_slice = self._bounds_tczyx

        # Convert slices to ranges for TCZ iteration
        t_start, t_stop, _ = t_slice.indices(self._meta.shape.t)
        c_start, c_stop, _ = c_slice.indices(self._meta.shape.c)
        z_start, z_stop, _ = z_slice.indices(self._meta.shape.z)

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
        """
        Implement numpy array protocol - materializes data from disk.

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
            output = output.astype(dtype)

        # NumPy 2.0+ copy parameter - data is always fresh from disk so no copy needed
        if copy and output is output:  # Always true, but satisfies the API
            output = output.copy()

        return output

    def __getitem__(self, key: Any) -> LazyBioArray:
        """
        Index the array with numpy-style syntax, returning a lazy view.

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
        """
        Normalize indexing key to tuple of slices/ints.

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

    def _parse_selection(
        self, key: tuple[slice | int, ...]
    ) -> tuple[range, range, range, slice, slice]:
        """
        Parse normalized key into ranges and slices.

        Returns
        -------
        tuple
            (t_range, c_range, z_range, y_slice, x_slice)
            Ranges are for iteration, slices are passed to read_plane
        """
        # Separate TCZYX dimensions (ignore RGB if present)
        t_key, c_key, z_key, y_key, x_key = key[:5]

        # Convert each dimension to range or slice
        t_range = self._key_to_range(t_key, self.shape[0])
        c_range = self._key_to_range(c_key, self.shape[1])
        z_range = self._key_to_range(z_key, self.shape[2])

        # Y and X stay as slices for sub-region reading
        y_slice = self._key_to_slice(y_key, self.shape[3])
        x_slice = self._key_to_slice(x_key, self.shape[4])

        return t_range, c_range, z_range, y_slice, x_slice

    def _key_to_range(self, key: int | slice, size: int) -> range:
        """Convert int or slice to range for iteration."""
        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = size + key
            if key < 0 or key >= size:
                msg = f"index {key} is out of bounds for axis with size {size}"
                raise IndexError(msg)
            return range(key, key + 1)
        else:
            # It's a slice
            start, stop, step = key.indices(size)
            return range(start, stop, step)

    def _key_to_slice(self, key: int | slice, size: int) -> slice:
        """Convert int or slice to slice for sub-region reading."""
        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = size + key
            if key < 0 or key >= size:
                msg = f"index {key} is out of bounds for axis with size {size}"
                raise IndexError(msg)
            return slice(key, key + 1)
        else:
            # Already a slice, validate bounds
            start, stop, _ = key.indices(size)
            return slice(start, stop)

    def _compute_output_shape(
        self, selection: tuple[range, range, range, slice, slice], key: tuple
    ) -> tuple[tuple[int, ...], list[bool]]:
        """
        Compute the shape of the output array and track squeezed dimensions.

        Returns
        -------
        tuple
            (output_shape, squeezed_dims) where squeezed_dims is a boolean list
            indicating which TCZYX dimensions were squeezed (True = squeezed)
        """
        t_range, c_range, z_range, y_slice, x_slice = selection

        # Track which dimensions are squeezed
        squeezed = [
            isinstance(key[0], int),  # T
            isinstance(key[1], int),  # C
            isinstance(key[2], int),  # Z
            isinstance(key[3], int),  # Y
            isinstance(key[4], int),  # X
        ]

        # Base shape from TCZYX
        shape = []

        # Add non-squeezed dimensions
        if not squeezed[0]:
            shape.append(len(t_range))
        if not squeezed[1]:
            shape.append(len(c_range))
        if not squeezed[2]:
            shape.append(len(z_range))
        if not squeezed[3]:
            y_start, y_stop, _ = y_slice.indices(self.shape[3])
            shape.append(y_stop - y_start)
        if not squeezed[4]:
            x_start, x_stop, _ = x_slice.indices(self.shape[4])
            shape.append(x_stop - x_start)

        # Handle RGB dimension if present
        if self.ndim == 6:
            if len(key) == 6 and not isinstance(key[5], int):
                shape.append(self.shape[5])
            elif len(key) < 6:
                # RGB dimension not indexed, keep it
                shape.append(self.shape[5])

        return tuple(shape), squeezed

    def _fill_output(
        self,
        output: np.ndarray,
        selection: tuple[range, range, range, slice, slice],
        squeezed: list[bool] | tuple[bool, bool, bool, bool, bool],
    ) -> None:
        """
        Fill output array by reading planes from Bio-Formats.

        Optimized to acquire lock once and reuse buffer across all reads.

        Parameters
        ----------
        output : np.ndarray
            Pre-allocated output array to fill
        selection : tuple
            (t_range, c_range, z_range, y_slice, x_slice) from _parse_selection
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

        # Acquire lock ONCE for entire batch read
        # This is much faster than acquiring/releasing on every plane
        with self._biofile._lock:
            # Set series and resolution once at start (not on every iteration)
            reader = self._biofile._ensure_java_reader()
            reader.setSeries(self._series)
            reader.setResolution(self._resolution)

            # Get metadata once (avoid repeated lookups in hot loop)
            meta = self._biofile.core_metadata(self._series, self._resolution)

            # Fast loop - no locking overhead, no validation, minimal copying!
            # Uses optimized _read_plane that skips all per-call overhead
            read_plane = self._biofile._read_plane  # Local reference for speed
            out_t = 0
            for t in t_range:
                out_c = 0
                for c in c_range:
                    out_z = 0
                    for z in z_range:
                        plane = read_plane(reader, meta, t, c, z, y_slice, x_slice)
                        if self._rgb_index is not None:
                            plane = plane[..., self._rgb_index]

                        # Build index tuple based on which dimensions are not squeezed
                        idx = []
                        if not squeezed[0]:  # T not squeezed
                            idx.append(out_t)
                        if not squeezed[1]:  # C not squeezed
                            idx.append(out_c)
                        if not squeezed[2]:  # Z not squeezed
                            idx.append(out_z)

                        # Assign plane to output (single copy: view → output)
                        if idx:
                            output[tuple(idx)] = plane
                        else:
                            # All T, C, Z squeezed - direct assignment
                            output[:] = plane

                        out_z += 1
                    out_c += 1
                out_t += 1
