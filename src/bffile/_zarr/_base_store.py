"""Read-only zarr v3 store base."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from typing_extensions import Self
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


class ReadOnlyStore(Store):
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range) for key, byte_range in key_ranges
        ]

    async def set(self, key: str, value: Buffer) -> None:
        raise PermissionError(f"{type(self).__name__} is read-only")

    async def delete(self, key: str) -> None:
        raise PermissionError(f"{type(self).__name__} is read-only")

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

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
            if remainder and remainder[0] == "/":
                remainder = remainder[1:]
            child = remainder.split("/")[0] if remainder else ""
            if child and child not in seen:
                seen.add(child)
                yield child

    async def _close(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # These are removed from the Store ABC ... just here in the off chance that someone
    # installs zarr 3.1

    def set_partial_values(  # pragma: no cover
        self,
        prototype: BufferPrototype,
        key_value_ranges: Iterable[tuple[str, Buffer, ByteRequest | None]],
    ) -> AsyncIterator[None]:
        raise PermissionError(f"{type(self).__name__} is read-only")

    @property
    def supports_partial_writes(self) -> Literal[False]:  # pragma: no cover
        return False

    @staticmethod
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
