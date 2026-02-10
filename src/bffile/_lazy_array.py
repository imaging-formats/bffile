"""Lazy numpy-compatible array for on-demand Bio-Formats reading."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bffile._biofile import BioFile


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
    Data is read from disk only when indexed, enabling efficient access to
    large datasets without loading everything into memory.

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

    Examples
    --------
    >>> with BioFile("image.nd2") as bf:
    ...     arr = bf.as_array()  # No data read yet
    ...     plane = arr[0, 0, 2]  # Read single plane
    ...     roi = arr[:, :, :, 100:200, 50:150]  # Read sub-region
    ...     full_data = np.array(arr)  # Materialize all data
    ...     max_z = np.max(arr, axis=2)  # Works with numpy functions

    Notes
    -----
    - BioFile must remain open while using this array
    - Step indexing (`arr[::2]`), fancy indexing, and boolean masks not supported
    - Not thread-safe: create separate BioFile instances per thread
    """

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
        meta = biofile.core_meta(series, resolution)

        # Follow same logic as to_numpy(): only include RGB dimension if > 1
        full_shape = meta.shape
        nt, nc, nz, ny, nx = full_shape[:5]
        nrgb = full_shape[5] if len(full_shape) == 6 else 1
        if nrgb > 1:
            self._shape = (nt, nc, nz, ny, nx, nrgb)
        else:
            self._shape = (nt, nc, nz, ny, nx)
        self._dtype = meta.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb) format."""
        return self._shape
    
    @property
    def size(self) -> int:
        """Number of elements in the array"""
        return math.prod(self._shape)

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
        Implement numpy array protocol.

        This enables `np.array(lazy_arr)` and other numpy operations.

        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type
        copy : bool, optional
            Whether to force a copy (NumPy 2.0+ compatibility)
        """
        # Read all data using our optimized __getitem__ implementation
        # (uses buffer reuse and batched locking)
        result = self[:]
        if dtype is not None and result.dtype != dtype:
            result = result.astype(dtype)
        # NumPy 2.0+ copy parameter - data is always fresh from disk so no copy needed
        if copy and result is result:  # Always true, but satisfies the API
            result = result.copy()
        return result

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Index the array with numpy-style syntax.

        Supports integer and slice indexing. Only reads the necessary planes
        and sub-regions from disk.

        Parameters
        ----------
        key : int, slice, tuple, or Ellipsis
            Index specification

        Returns
        -------
        np.ndarray
            The requested data

        Raises
        ------
        NotImplementedError
            If fancy indexing, boolean indexing, or step != 1 is used
        IndexError
            If indices are out of bounds
        """
        # Normalize key to tuple
        key = self._normalize_key(key)

        # Parse selection into (t_range, c_range, z_range, y_slice, x_slice)
        selection = self._parse_selection(key)

        # Compute output shape and track squeezed dimensions
        output_shape, squeezed = self._compute_output_shape(selection, key)

        # Allocate output array
        output = np.empty(output_shape, dtype=self.dtype)

        # Fill output by reading planes
        self._fill_output(output, selection, squeezed)

        return output

    def _normalize_key(self, key: Any) -> tuple[slice | int, ...]:
        """
        Normalize indexing key to tuple of slices/ints.

        Handles scalars, tuples, ellipsis expansion, and RGB dimension.
        """
        # Convert scalar to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Check for unsupported indexing types FIRST (before ellipsis check)
        for _i, k in enumerate(key):
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
        squeezed: list[bool],
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
        y_start, y_stop, _ = y_slice.indices(self.shape[3])
        x_start, x_stop, _ = x_slice.indices(self.shape[4])
        if y_stop <= y_start or x_stop <= x_start:
            return

        # Acquire lock ONCE for entire batch read
        # This is much faster than acquiring/releasing on every plane
        with self._biofile._lock:
            # Set series and resolution once at start (not on every iteration)
            reader = self._biofile.java_reader()
            reader.setSeries(self._series)
            reader.setResolution(self._resolution)

            # Get metadata once (avoid repeated lookups in hot loop)
            meta = self._biofile.core_meta(self._series, self._resolution)

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
