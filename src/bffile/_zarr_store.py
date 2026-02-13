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


class BioFormatsStore(Store):
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
    memoize : int or bool, optional
        Memoizer threshold in ms (only used in standalone mode), by default 0.

    Examples
    --------
    From a LazyBioArray (borrowed BioFile):

    >>> with BioFile("image.nd2") as bf:
    ...     store = bf.as_array().as_zarr()
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
    ) -> None:
        super().__init__(read_only=True)

        from bffile._lazy_array import LazyBioArray

        if not isinstance(obj, LazyBioArray):
            raise TypeError("BioFormatsStore requires a LazyBioArray as input")
        self._lazy_array = obj
        self._meta = obj._meta
        self._tile_size = tile_size
        self._metadata_bytes: bytes | None = None
        self._chunk_keys: set[str] | None = None
        self._is_open = True

    @property
    def _biofile(self) -> BioFile:
        return self._lazy_array._biofile

    # ------------------------------------------------------------------
    # Metadata & chunk key helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> bytes:
        """Build and cache zarr.json metadata as bytes."""
        if self._metadata_bytes is not None:
            return self._metadata_bytes

        meta = self._meta
        shape = meta.shape

        array_shape = list(shape.as_array_shape)

        if self._tile_size is not None:
            ty, tx = self._tile_size
            chunks = [1, 1, 1, ty, tx]
        else:
            chunks = [1, 1, 1, shape.y, shape.x]
        if shape.rgb > 1:
            chunks.append(shape.rgb)

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
        """Build and cache the set of valid chunk keys."""
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

        keys: set[str] = set()
        rgb_suffix = "/0" if shape.rgb > 1 else ""
        for t in range(shape.t):
            for c in range(shape.c):
                for z in range(shape.z):
                    for yi in range(ny):
                        for xi in range(nx):
                            keys.add(f"c/{t}/{c}/{z}/{yi}/{xi}{rgb_suffix}")

        self._chunk_keys = keys
        return keys

    def _read_chunk(self, key: str) -> bytes:
        """Read a chunk by its key and return raw bytes."""
        parts = key.split("/")
        # key format: "c/<t>/<c>/<z>/<yi>/<xi>" or "c/<t>/<c>/<z>/<yi>/<xi>/0"
        t, c, z, yi, xi = (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
        )

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
            c=c,
            z=z,
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
            series=arr._series,
            resolution=arr._resolution,
        )

        # Pad edge chunks to full tile size (zarr expects full chunk shape)
        if self._tile_size is not None:
            ty, tx = self._tile_size
            actual_h = y_stop - y_start
            actual_w = x_stop - x_start
            if actual_h < ty or actual_w < tx:
                if shape.rgb > 1:
                    padded = np.zeros((ty, tx, shape.rgb), dtype=self._meta.dtype)
                else:
                    padded = np.zeros((ty, tx), dtype=self._meta.dtype)
                padded[:actual_h, :actual_w] = plane
                plane = padded

        # Ensure contiguous C-order for correct byte layout
        return np.ascontiguousarray(plane).tobytes()

    # ------------------------------------------------------------------
    # Store ABC — required abstract methods
    # ------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BioFormatsStore):
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

    def __enter__(self) -> BioFormatsStore:
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
