"""Read-only zarr v3 group store backed by Bio-Formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ._utils import physical_pixel_sizes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from ome_types import OME
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.storage import StoreLike

    from bffile._biofile import BioFile
    from bffile._core_metadata import CoreMetadata
    from bffile._zarr_store import BioFormatsStore


class PathLevel(Enum):
    """Classification of path levels in the zarr hierarchy."""

    ROOT = auto()
    OME_GROUP = auto()
    OME_METADATA = auto()
    SERIES_GROUP = auto()
    MULTISCALES_GROUP = auto()
    ARRAY_METADATA = auto()
    CHUNK = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ParsedPath:
    """Structured information parsed from a zarr path."""

    level: PathLevel
    series: int | None = None
    resolution: int | None = None
    chunk_key: str | None = None


class PathRouter:
    """Parses hierarchical zarr keys into structured information."""

    @staticmethod
    def parse(key: str) -> ParsedPath:
        """Parse a zarr key and return structured path information.

        Path patterns (NGFF v0.5):
        - zarr.json → root group metadata
        - OME/zarr.json → OME group metadata
        - OME/METADATA.ome.xml → OME-XML string
        - {series}/zarr.json → series/multiscales group metadata
        - {series}/{resolution}/zarr.json → array metadata
        - {series}/{resolution}/c/... → chunk data

        Parameters
        ----------
        key : str
            The zarr key to parse.

        Returns
        -------
        ParsedPath
            Structured information about the path.
        """
        if key == "zarr.json":
            return ParsedPath(level=PathLevel.ROOT)

        if key == "OME/zarr.json":
            return ParsedPath(level=PathLevel.OME_GROUP)

        if key == "OME/METADATA.ome.xml":
            return ParsedPath(level=PathLevel.OME_METADATA)

        parts = key.split("/")

        # Series/multiscales group: {series}/zarr.json
        if len(parts) == 2 and parts[1] == "zarr.json":
            try:
                series = int(parts[0])
                return ParsedPath(level=PathLevel.MULTISCALES_GROUP, series=series)
            except ValueError:
                return ParsedPath(level=PathLevel.UNKNOWN)

        # Array metadata: {series}/{resolution}/zarr.json
        if len(parts) == 3 and parts[2] == "zarr.json":
            try:
                series = int(parts[0])
                resolution = int(parts[1])
                return ParsedPath(
                    level=PathLevel.ARRAY_METADATA,
                    series=series,
                    resolution=resolution,
                )
            except ValueError:
                return ParsedPath(level=PathLevel.UNKNOWN)

        # Chunk data: {series}/{resolution}/c/...
        if len(parts) >= 4 and parts[2] == "c":
            try:
                series = int(parts[0])
                resolution = int(parts[1])
                # Reconstruct chunk key: "c/t/c/z/yi/xi" or "c/t/c/z/yi/xi/0"
                chunk_key = "/".join(parts[2:])
                return ParsedPath(
                    level=PathLevel.CHUNK,
                    series=series,
                    resolution=resolution,
                    chunk_key=chunk_key,
                )
            except ValueError:
                return ParsedPath(level=PathLevel.UNKNOWN)

        return ParsedPath(level=PathLevel.UNKNOWN)


class BioFormatsGroupStore(Store):
    """Read-only zarr v3 group store for complete Bio-Formats file hierarchy.

    Virtualizes an entire Bio-Formats file as an OME-ZARR group containing
    all series and resolution levels, following NGFF v0.5 specification.

    Directory structure:
        root/
        ├── zarr.json (group metadata with bioformats2raw.layout: 3)
        ├── OME/
        │   ├── zarr.json (group with series list)
        │   └── METADATA.ome.xml (raw OME-XML)
        ├── 0/ (series 0 - multiscales group)
        │   ├── zarr.json (multiscales metadata with ome.version=0.5)
        │   ├── 0/ (full resolution)
        │   │   ├── zarr.json (array)
        │   │   └── c/... (chunks)
        │   └── 1/ (downsampled, if exists)
        └── 1/ (series 1, if exists)

    Parameters
    ----------
    biofile : BioFile
        An open BioFile instance. Must remain open for the lifetime of the store.
    tile_size : tuple[int, int], optional
        If provided, Y and X are chunked into tiles of this size.

    Examples
    --------
    >>> with BioFile("image.nd2") as bf:
    ...     group = zarr.open_group(bf.as_zarr_group(), mode="r")
    ...     # Access first series, full resolution
    ...     arr = group["0/0"]
    ...     data = arr[0, 0, 0]
    """

    def __init__(
        self,
        biofile: BioFile,
        /,
        *,
        tile_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self._biofile = biofile
        self._tile_size = tile_size
        self._array_stores: dict[tuple[int, int], BioFormatsStore] = {}
        self._is_open = True

    # ------------------------------------------------------------------
    # Metadata builders
    # ------------------------------------------------------------------

    def _build_root_metadata(self) -> bytes:
        """Build root group metadata with bioformats2raw layout marker (NGFF v0.5)."""
        metadata: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "bioformats2raw.layout": 3,
                },
            },
        }
        return json.dumps(metadata).encode()

    def _build_ome_metadata(self) -> bytes:
        """Build OME group metadata with series list (NGFF v0.5)."""
        series_count = len(self._biofile)
        metadata: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "series": [str(i) for i in range(series_count)],
                }
            },
        }
        return json.dumps(metadata).encode()

    def _build_multiscales_metadata(self, series: int) -> bytes:
        """Build multiscales group metadata with axes and datasets (NGFF v0.5).

        In NGFF v0.5, the series group IS the multiscales group.
        This includes:
        - axes: Dimension information with types and units
        - datasets: List of resolution levels with coordinate transforms
        """
        meta = self._biofile[series].core_metadata()
        ome = self._biofile.ome_metadata

        # NGFF v0.5 multiscales are limited to 5 dimensions
        # RGB images (rgb > 1) create 6D arrays which are not supported
        if meta.shape.rgb > 1:
            raise NotImplementedError("RGB not yet implemented")

        # Build axes list (order: T, C, Z, Y, X)
        axes = self._build_axes(meta, ome, series)

        # Build datasets list (one per resolution)
        datasets = self._build_datasets(meta, ome, series)

        metadata: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "multiscales": [
                        {
                            "version": "0.5",
                            "name": f"Series {series}",
                            "axes": axes,
                            "datasets": datasets,
                        }
                    ],
                }
            },
        }
        return json.dumps(metadata).encode()

    def _build_axes(
        self, meta: CoreMetadata, ome: OME, series: int
    ) -> list[dict[str, str]]:
        """Build axes list from metadata.

        Order: T, C, Z, Y, X, [RGB] (standard OME-ZARR)
        Include type (time/channel/space) and units where appropriate.

        Note: Always include all axes to match array dimensions.
        Arrays are 5D (TCZYX) or 6D (TCZYXS) for RGB.
        """
        axes: list[dict[str, str]] = []

        # Time axis (always include)
        axis: dict[str, str] = {"name": "t", "type": "time"}
        # Try to get time unit from OME
        if ome.images[series].pixels.time_increment is not None:
            axis["unit"] = "millisecond"
        axes.append(axis)

        # Channel axis (always include)
        axes.append({"name": "c", "type": "channel"})

        # Spatial axes (always include)
        for name in ["z", "y", "x"]:
            axis = {"name": name, "type": "space"}
            # Add units if available
            physical_size = getattr(ome.images[series].pixels, f"physical_size_{name}")
            if physical_size is not None:
                axis["unit"] = "micrometer"
            axes.append(axis)

        # RGB/samples axis (only for RGB images with rgb > 1)
        if meta.shape.rgb > 1:
            axes.append({"name": "s", "type": "channel"})

        return axes

    def _build_datasets(
        self, meta: CoreMetadata, ome: OME, series: int
    ) -> list[dict[str, Any]]:
        """Build datasets list with coordinate transforms for each resolution."""
        datasets: list[dict[str, Any]] = []

        # Get physical pixel sizes
        pps = physical_pixel_sizes(ome, series)

        for res in range(meta.resolution_count):
            # Dataset path
            dataset: dict[str, Any] = {"path": str(res)}

            # Build coordinate transforms (scale values)
            # Include all dimensions to match array shape (5D or 6D for RGB)
            scale = [
                1.0,  # Time (no scaling)
                1.0,  # Channel (no scaling)
                pps.z if pps.z is not None else 1.0,  # Z spatial scale
                pps.y if pps.y is not None else 1.0,  # Y spatial scale
                pps.x if pps.x is not None else 1.0,  # X spatial scale
            ]

            # Add RGB/samples dimension if present
            if meta.shape.rgb > 1:
                scale.append(1.0)  # RGB samples (no scaling)

            # TODO: For resolution > 0, scale factors should account for downsampling
            # For now, we just use the same scale for all resolutions
            # In the future, we should query the actual pixel sizes per resolution

            dataset["coordinateTransformations"] = [{"type": "scale", "scale": scale}]
            datasets.append(dataset)

        return datasets

    def _get_array_store(self, series: int, resolution: int) -> BioFormatsStore:
        """Get or create cached array store for a series/resolution."""
        key = (series, resolution)
        if key not in self._array_stores:
            arr = self._biofile.as_array(series, resolution)
            self._array_stores[key] = arr.zarr_store(tile_size=self._tile_size)
        return self._array_stores[key]

    # ------------------------------------------------------------------
    # Store ABC — required abstract methods
    # ------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BioFormatsGroupStore):
            return NotImplemented
        return (
            self._biofile.filename == value._biofile.filename
            and self._tile_size == value._tile_size
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Get data for a key in the zarr hierarchy."""
        parsed = PathRouter.parse(key)

        # Route to appropriate handler
        if parsed.level == PathLevel.ROOT:
            data = self._build_root_metadata()
        elif parsed.level == PathLevel.OME_GROUP:
            data = self._build_ome_metadata()
        elif parsed.level == PathLevel.OME_METADATA:
            # Return raw OME-XML string
            data = self._biofile.ome_xml.encode()
        elif parsed.level == PathLevel.MULTISCALES_GROUP:
            # In NGFF v0.5, series group IS the multiscales group
            data = self._build_multiscales_metadata(parsed.series)  # type: ignore[arg-type]
        elif parsed.level == PathLevel.ARRAY_METADATA:
            # Delegate to array store
            store = self._get_array_store(parsed.series, parsed.resolution)  # type: ignore[arg-type]
            return await store.get("zarr.json", prototype, byte_range)
        elif parsed.level == PathLevel.CHUNK:
            # Delegate to array store
            store = self._get_array_store(parsed.series, parsed.resolution)  # type: ignore[arg-type]
            with self._biofile.ensure_open():
                return await store.get(parsed.chunk_key, prototype, byte_range)  # type: ignore[arg-type]
        else:
            return None

        # Apply byte range if requested
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
        """Check if a key exists in the hierarchy."""
        parsed = PathRouter.parse(key)

        if parsed.level == PathLevel.UNKNOWN:
            return False

        # Root, OME group, and OME metadata always exist
        if parsed.level in (
            PathLevel.ROOT,
            PathLevel.OME_GROUP,
            PathLevel.OME_METADATA,
        ):
            return True

        # Multiscales group (series group): check series index is valid
        if parsed.level == PathLevel.MULTISCALES_GROUP:
            return 0 <= parsed.series < len(self._biofile)  # type: ignore[operator]

        # Array metadata: check series and resolution are valid
        if parsed.level == PathLevel.ARRAY_METADATA:
            if not (0 <= parsed.series < len(self._biofile)):  # type: ignore[operator]
                return False
            meta = self._biofile[parsed.series].core_metadata()  # type: ignore[index]
            return 0 <= parsed.resolution < meta.resolution_count  # type: ignore[operator]

        # Chunk: delegate to array store
        if parsed.level == PathLevel.CHUNK:
            if not (0 <= parsed.series < len(self._biofile)):  # type: ignore[operator]
                return False
            meta = self._biofile[parsed.series].core_metadata()  # type: ignore[index]
            if not (0 <= parsed.resolution < meta.resolution_count):  # type: ignore[operator]
                return False
            store = self._get_array_store(parsed.series, parsed.resolution)  # type: ignore[arg-type]
            return await store.exists(parsed.chunk_key)  # type: ignore[arg-type]

        return False

    async def set(self, key: str, value: Buffer) -> None:
        raise PermissionError("BioFormatsGroupStore is read-only")

    async def delete(self, key: str) -> None:
        raise PermissionError("BioFormatsGroupStore is read-only")

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
        """List all valid keys in the hierarchy."""
        # Root metadata
        yield "zarr.json"

        # OME group
        yield "OME/zarr.json"
        yield "OME/METADATA.ome.xml"

        # Enumerate all series
        for series_idx in range(len(self._biofile)):
            # Series/multiscales group (NGFF v0.5: series IS multiscales)
            yield f"{series_idx}/zarr.json"

            # Get metadata for this series
            meta = self._biofile[series_idx].core_metadata()

            # Enumerate all resolutions
            for res_idx in range(meta.resolution_count):
                # Array metadata
                yield f"{series_idx}/{res_idx}/zarr.json"

                # Chunks - delegate to array store
                store = self._get_array_store(series_idx, res_idx)
                async for chunk_key in store.list():
                    if chunk_key != "zarr.json":
                        yield f"{series_idx}/{res_idx}/{chunk_key}"

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the store and all cached array stores."""
        self._is_open = False
        for store in self._array_stores.values():
            store.close()
        self._array_stores.clear()

    async def _close(self) -> None:
        self.close()

    def __enter__(self) -> BioFormatsGroupStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"BioFormatsGroupStore({self._biofile.filename!r}, "
            f"series_count={len(self._biofile)})"
        )

    # ------------------------------------------------------------------
    # These are removed from the Store ABC ... just here in the off chance that someone
    # installs zarr 3.1

    def set_partial_values(  # pragma: no cover
        self,
        prototype: BufferPrototype,
        key_value_ranges: Iterable[tuple[str, Buffer, ByteRequest | None]],
    ) -> AsyncIterator[None]:
        raise PermissionError("BioFormatsGroupStore is read-only")

    @property
    def supports_partial_writes(self) -> Literal[False]:  # pragma: no cover
        return False

    async def _copy_to(self, dest: Store) -> None:
        proto = default_buffer_prototype()
        async for key in self.list():
            buf = await self.get(key, prototype=proto)
            if buf is not None:
                await dest.set(key, buf)

    def save(self, dest: StoreLike) -> None:
        """Save the store contents to the given `dest`.

        Parameters
        ----------
        dest : zarr.storage.StoreLike
            A zarr-compatible store to which the contents of this group store will be
            copied. This can be a path string, a zarr.storage.Store instance, or any
            object accepted by zarr.open_group().
        """
        group = zarr.open_group(dest, mode="w")
        sync(self._copy_to(group.store))


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
