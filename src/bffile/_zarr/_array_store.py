"""Read-only zarr v3 store backed by Bio-Formats."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.core.buffer import Buffer, BufferPrototype

    from bffile._biofile import BioFile
    from bffile._lazy_array import LazyBioArray


class BFArrayStore(Store):
    """Read-only zarr v3 store that virtualizes a Bio-Formats file.

    Each zarr chunk maps to a single ``read_plane()`` call, producing raw
    uncompressed plane bytes on demand. No data is written to disk.

    Parameters
    ----------
    path_or_lazy_array : str, PathLike, or LazyBioArray
        Either a file path (standalone mode, store owns its BioFile) or a
        LazyBioArray (borrowed mode, caller manages the BioFile lifecycle).
    tile_size : tuple[int, int], optional
        If provided, Y and X are chunked into tiles of this size instead of
        full planes. Chunk shape becomes ``(1, 1, 1, tile_y, tile_x)``.
    rgb_as_channels : bool, optional
        If True, interleave RGB samples as separate C channels (OME-Zarr convention).
        If False, keep RGB as the last dimension (numpy/imread convention).
        Default is False.
    squeeze_singletons : bool, optional
        If True, omit dimensions with size 1 from array shape (except Y/X).
        Default is False.

    Examples
    --------
    From a LazyBioArray (borrowed BioFile):

    >>> with BioFile("image.nd2") as bf:
    ...     store = bf.as_array().zarr_store()
    ...     arr = zarr.open(store, mode="r")
    ...     data = arr[0, 0, 0]
    """

    def __init__(
        self,
        obj: LazyBioArray,
        /,
        *,
        tile_size: tuple[int, int] | None = None,
        memoize: int | bool = 0,
        rgb_as_channels: bool = False,
        squeeze_singletons: bool = False,
    ) -> None:
        super().__init__(read_only=True)

        from bffile._lazy_array import LazyBioArray

        if not isinstance(obj, LazyBioArray):
            raise TypeError("BioFormatsStore requires a LazyBioArray as input")
        self._lazy_array = obj
        self._meta = obj._meta
        self._tile_size = tile_size
        self._rgb_as_channels = rgb_as_channels
        self._squeeze_singletons = squeeze_singletons
        self._metadata_bytes: bytes | None = None
        self._chunk_keys: set[str] | None = None
        self._is_open = True

        # Pre-compute effective shape and dimension filter
        self._effective_shape = self._compute_effective_shape()
        self._dim_filter = self._compute_dimension_filter()

    @property
    def _biofile(self) -> BioFile:
        return self._lazy_array._biofile

    def _compute_effective_shape(self) -> tuple[int, ...]:
        """Compute effective shape with RGB handling and dimension squeezing."""
        shape = self._meta.shape
        t, c, z, y, x, rgb = shape.t, shape.c, shape.z, shape.y, shape.x, shape.rgb

        if self._rgb_as_channels:
            # Interleave RGB samples as separate C channels
            full_shape = [t, c * rgb, z, y, x]
        else:
            # Keep RGB as last dimension (matching imread behavior)
            if rgb > 1:
                full_shape = [t, c, z, y, x, rgb]
            else:
                full_shape = [t, c, z, y, x]

        # Dimension squeezing: omit singletons (except Y/X and RGB always kept)
        if self._squeeze_singletons:
            dim_filter = self._compute_dimension_filter()
            return tuple(
                s for s, keep in zip(full_shape, dim_filter, strict=False) if keep
            )

        return tuple(full_shape)

    def _compute_dimension_filter(self) -> list[bool]:
        """Return mask indicating which dims to include.

        Returns [T, C, Z, Y, X] for 5D arrays or [T, C, Z, Y, X, RGB] for 6D arrays.
        """
        shape = self._meta.shape

        if self._rgb_as_channels:
            # RGB interleaved into C: always 5D
            if not self._squeeze_singletons:
                return [True, True, True, True, True]
            c_eff = shape.c * shape.rgb
            return [
                shape.t > 1,  # T
                c_eff > 1,  # C (with RGB interleaved)
                shape.z > 1,  # Z
                True,  # Y (always keep)
                True,  # X (always keep)
            ]
        else:
            # RGB as separate dimension: 5D or 6D
            has_rgb_dim = shape.rgb > 1
            if not self._squeeze_singletons:
                base_filter = [True, True, True, True, True]
                if has_rgb_dim:
                    base_filter.append(True)  # RGB (always keep if present)
                return base_filter

            base_filter = [
                shape.t > 1,  # T
                shape.c > 1,  # C
                shape.z > 1,  # Z
                True,  # Y (always keep)
                True,  # X (always keep)
            ]
            if has_rgb_dim:
                base_filter.append(True)  # RGB (always keep if present)
            return base_filter

    # ------------------------------------------------------------------
    # Metadata & chunk key helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> bytes:
        """Build and cache zarr.json metadata as bytes."""
        if self._metadata_bytes is not None:
            return self._metadata_bytes

        meta = self._meta
        shape = meta.shape

        # Use effective shape (RGB handled, singletons squeezed)
        array_shape = list(self._effective_shape)

        # Build chunk shape to match effective shape
        if self._tile_size is not None:
            ty, tx = self._tile_size
            base_chunks = [1, 1, 1, ty, tx]
        else:
            base_chunks = [1, 1, 1, shape.y, shape.x]

        # Add RGB dimension to chunk if present in array (never chunked separately)
        if not self._rgb_as_channels and shape.rgb > 1:
            base_chunks.append(shape.rgb)

        # Apply dimension filter to get final chunk shape
        chunks = [
            c for c, keep in zip(base_chunks, self._dim_filter, strict=False) if keep
        ]

        endian = "little" if meta.is_little_endian else "big"

        metadata: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": array_shape,
            "data_type": np.dtype(meta.dtype).name,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunks},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": endian}},
            ],
        }

        self._metadata_bytes = json.dumps(metadata).encode()
        return self._metadata_bytes

    def _get_chunk_keys(self) -> set[str]:
        """Build and cache chunk keys for effective shape.

        Keys match array dimensionality (5D or 6D). For 6D arrays, RGB
        coordinate is always 0 since RGB is never chunked separately.
        """
        if self._chunk_keys is not None:
            return self._chunk_keys

        shape = self._meta.shape

        if self._tile_size is not None:
            ty, tx = self._tile_size
            ny = math.ceil(shape.y / ty)
            nx = math.ceil(shape.x / tx)
        else:
            ny = 1
            nx = 1

        # Generate keys for effective shape (RGB handled, singletons squeezed)
        keys: set[str] = set()

        # Build ranges for each dimension
        t_range = range(shape.t)
        if self._rgb_as_channels:
            c_range = range(shape.c * shape.rgb)
        else:
            c_range = range(shape.c)
        z_range = range(shape.z)
        yi_range = range(ny)
        xi_range = range(nx)

        # Build ranges list (5D or 6D depending on RGB mode)
        ranges = [t_range, c_range, z_range, yi_range, xi_range]

        # Add RGB range if present (always range(1) - never chunk along RGB)
        if not self._rgb_as_channels and shape.rgb > 1:
            ranges.append(range(1))  # Single chunk for RGB dimension

        # Apply dimension filter to generate squeezed keys
        active_ranges = [
            r for r, keep in zip(ranges, self._dim_filter, strict=False) if keep
        ]

        import itertools

        for indices in itertools.product(*active_ranges):
            keys.add(f"c/{'/'.join(str(i) for i in indices)}")

        self._chunk_keys = keys
        return keys

    def _parse_chunk_key(self, key: str) -> tuple[int, int, int, int, int]:
        """Parse squeezed chunk key to (t, c, z, yi, xi) coordinates.

        Note: Returns 5D TCZYX coordinates. If key includes RGB coordinate
        (for 6D arrays), it's ignored since RGB is always 0.
        """
        parts = key.split("/")
        squeezed_indices = [int(p) for p in parts[1:]]  # Skip "c" prefix

        # Map squeezed indices back to full dimensions (5D or 6D)
        # We always return 5D (TCZYX), ignoring RGB coordinate if present
        has_rgb_dim = not self._rgb_as_channels and self._meta.shape.rgb > 1
        ndims = 6 if has_rgb_dim else 5

        full_indices = [0] * ndims  # [t, c, z, yi, xi] or [t, c, z, yi, xi, rgb]
        kept_dims = [i for i, keep in enumerate(self._dim_filter) if keep]

        for squeezed_i, dim_i in enumerate(kept_dims):
            full_indices[dim_i] = squeezed_indices[squeezed_i]

        # Return only TCZYX (first 5), dropping RGB coordinate if present
        return tuple(full_indices[:5])  # type: ignore[return-value]

    def _read_chunk(self, key: str) -> bytes:
        """Read a chunk by its key and return raw bytes."""
        # Parse key (handles both squeezed and full formats)
        t, c_eff, z, yi, xi = self._parse_chunk_key(key)

        # Map effective C back to base (c, rgb_sample) if RGB interleaved as channels
        if self._rgb_as_channels:
            c_base = c_eff // self._meta.shape.rgb
            rgb_sample = c_eff % self._meta.shape.rgb
        else:
            c_base = c_eff
            rgb_sample = None

        arr = self._lazy_array
        shape = self._meta.shape

        if self._tile_size is not None:
            ty, tx = self._tile_size
            y_start = yi * ty
            x_start = xi * tx
            y_stop = min(y_start + ty, shape.y)
            x_stop = min(x_start + tx, shape.x)
        else:
            y_start, y_stop = 0, shape.y
            x_start, x_stop = 0, shape.x

        plane = self._biofile.read_plane(
            t=t,
            c=c_base,  # Use base channel index
            z=z,
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
            series=arr._series,
            resolution=arr._resolution,
        )

        # Extract single RGB channel if interleaving RGB as channels
        # Only slice if the image actually has RGB channels (rgb > 1)
        if rgb_sample is not None and self._meta.shape.rgb > 1:
            plane = plane[..., rgb_sample]  # (Y, X, RGB) → (Y, X)

        # Pad edge chunks to full tile size (zarr expects full chunk shape)
        if self._tile_size is not None:
            ty, tx = self._tile_size
            actual_h = y_stop - y_start
            actual_w = x_stop - x_start
            if actual_h < ty or actual_w < tx:
                # Determine expected shape based on RGB mode
                if rgb_sample is not None or shape.rgb == 1:
                    # RGB extracted or no RGB: plane is 2D (Y, X)
                    padded = np.zeros((ty, tx), dtype=self._meta.dtype)
                    padded[:actual_h, :actual_w] = plane
                else:
                    # RGB kept: plane is 3D (Y, X, RGB)
                    padded = np.zeros((ty, tx, shape.rgb), dtype=self._meta.dtype)
                    padded[:actual_h, :actual_w, :] = plane
                plane = padded

        # Ensure contiguous C-order for correct byte layout
        return np.ascontiguousarray(plane).tobytes()

    # ------------------------------------------------------------------
    # Store ABC — required abstract methods
    # ------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BFArrayStore):
            return NotImplemented
        return (
            self._biofile.filename == value._biofile.filename
            and self._lazy_array._series == value._lazy_array._series
            and self._lazy_array._resolution == value._lazy_array._resolution
            and self._tile_size == value._tile_size
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if key == "zarr.json":
            data = self._build_metadata()
        elif key in self._get_chunk_keys():
            with self._biofile.ensure_open():
                data = self._read_chunk(key)
        else:
            return None

        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)

        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range) for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return key == "zarr.json" or key in self._get_chunk_keys()

    async def set(self, key: str, value: Buffer) -> None:
        raise PermissionError("BioFormatsStore is read-only")

    async def delete(self, key: str) -> None:
        raise PermissionError("BioFormatsStore is read-only")

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        yield "zarr.json"
        for key in self._get_chunk_keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        seen: set[str] = set()
        async for key in self.list():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            child = remainder.split("/")[0]
            if child and child not in seen:
                seen.add(child)
                yield child

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the store (and owned BioFile, if any)."""
        self._is_open = False

    async def _close(self) -> None:
        self.close()

    def __enter__(self) -> BFArrayStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        arr = self._lazy_array
        return (
            f"BioFormatsStore({self._biofile.filename!r}, "
            f"series={arr._series}, shape={arr.shape})"
        )

    # ------------------------------------------------------------------
    # These are removed from the Store ABC ... just here in the off chance that someone
    # installs zarr 3.1

    def set_partial_values(  # pragma: no cover
        self,
        prototype: BufferPrototype,
        key_value_ranges: Iterable[tuple[str, Buffer, ByteRequest | None]],
    ) -> AsyncIterator[None]:
        raise PermissionError("BioFormatsStore is read-only")

    @property
    def supports_partial_writes(self) -> Literal[False]:  # pragma: no cover
        return False


def _apply_byte_range(data: bytes, byte_range: ByteRequest) -> bytes:
    """Slice *data* according to a zarr ByteRequest."""
    n = len(data)
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[n - byte_range.suffix :]
    raise TypeError(f"Unexpected byte_range type: {type(byte_range)}")
