"""Read-only zarr v3 group store backed by Bio-Formats."""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple

import zarr
from ome_types.model import UnitsLength, UnitsTime
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from bffile._utils import physical_pixel_sizes
from bffile._zarr._base_store import ReadOnlyStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from ome_types import OME
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.storage import StoreLike

    from bffile._biofile import BioFile
    from bffile._zarr._array_store import BFArrayStore

# OME-NGFF dimension type mapping
_DIMENSION_TYPES = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "y": "space",
    "x": "space",
    "r": "channel",  # RGB (unused: group store uses rgb_as_channels=True)
}

# OME to NGFF unit mappings (only NGFF-whitelisted units)
_OME_TO_NGFF_TIME: dict[UnitsTime, str] = {
    UnitsTime.YOTTASECOND: "yottasecond",
    UnitsTime.ZETTASECOND: "zettasecond",
    UnitsTime.EXASECOND: "exasecond",
    UnitsTime.TERASECOND: "terasecond",
    UnitsTime.GIGASECOND: "gigasecond",
    UnitsTime.MEGASECOND: "megasecond",
    UnitsTime.KILOSECOND: "kilosecond",
    UnitsTime.HECTOSECOND: "hectosecond",
    UnitsTime.SECOND: "second",
    UnitsTime.DECISECOND: "decisecond",
    UnitsTime.CENTISECOND: "centisecond",
    UnitsTime.MILLISECOND: "millisecond",
    UnitsTime.MICROSECOND: "microsecond",
    UnitsTime.NANOSECOND: "nanosecond",
    UnitsTime.PICOSECOND: "picosecond",
    UnitsTime.FEMTOSECOND: "femtosecond",
    UnitsTime.ATTOSECOND: "attosecond",
    UnitsTime.ZEPTOSECOND: "zeptosecond",
    UnitsTime.MINUTE: "minute",
    UnitsTime.HOUR: "hour",
    UnitsTime.DAY: "day",
}

_OME_TO_NGFF_LENGTH: dict[UnitsLength, str] = {
    UnitsLength.YOTTAMETER: "yottameter",
    UnitsLength.ZETTAMETER: "zettameter",
    UnitsLength.EXAMETER: "exameter",
    UnitsLength.PETAMETER: "petameter",
    UnitsLength.TERAMETER: "terameter",
    UnitsLength.GIGAMETER: "gigameter",
    UnitsLength.MEGAMETER: "megameter",
    UnitsLength.KILOMETER: "kilometer",
    UnitsLength.HECTOMETER: "hectometer",
    UnitsLength.METER: "meter",
    UnitsLength.DECIMETER: "decimeter",
    UnitsLength.CENTIMETER: "centimeter",
    UnitsLength.MILLIMETER: "millimeter",
    UnitsLength.MICROMETER: "micrometer",
    UnitsLength.NANOMETER: "nanometer",
    UnitsLength.PICOMETER: "picometer",
    UnitsLength.FEMTOMETER: "femtometer",
    UnitsLength.ATTOMETER: "attometer",
    UnitsLength.ZEPTOMETER: "zeptometer",
    UnitsLength.ANGSTROM: "angstrom",
    UnitsLength.INCH: "inch",
    UnitsLength.FOOT: "foot",
}


class BFOmeZarrStore(ReadOnlyStore):
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
        self._array_stores: dict[tuple[int, int], BFArrayStore] = {}
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

        Note
        ----
        RGB images are currently represented as 6D arrays (TCZYXS) which is not
        strictly NGFF v0.5 compliant (spec recommends 5D with expanded C dimension).
        A future enhancement would wrap the Bio-Formats reader with ChannelSeparator
        to automatically split RGB into separate C channels, matching bioformats2raw
        behavior. For now, RGB images are accessible but may not be fully compliant
        with all NGFF tools.
        """
        meta = self._biofile.core_metadata(series=series)
        ome = self._biofile.ome_metadata
        datasets = self._build_datasets(ome, series, meta.resolution_count)

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
                            "axes": list(self._build_axes(ome, series)),
                            "datasets": datasets,
                        }
                    ],
                }
            },
        }
        return json.dumps(metadata).encode()

    def _build_axes(self, ome: OME, series: int) -> Iterator[dict[str, str]]:
        """Build axes list from metadata.

        Omits singleton dimensions (size 1) except Y and X which are always
        included per NGFF requirements. Delegates to array store's dimension
        filter for consistency with chunk generation.

        Units are mapped from OME-XML unit enums to NGFF v0.5 unit strings.

        Note
        ----
        For RGB images, the C dimension is expanded to include RGB samples
        (e.g., C=2 with RGB=3 becomes C=6).
        """
        pixels = ome.images[series].pixels
        store_0 = self._get_array_store(series, 0)
        for name in store_0.dimension_names():
            dim_type = _DIMENSION_TYPES.get(name, "other")
            axis: dict[str, str] = {"name": name, "type": dim_type}
            if (
                dim_type == "time"
                and "time_increment_unit" in pixels.model_fields_set
                and (unit := _OME_TO_NGFF_TIME.get(pixels.time_increment_unit))
            ):
                axis["unit"] = unit
            elif dim_type == "space":
                field_name = f"physical_size_{name}_unit"
                if field_name in pixels.model_fields_set:
                    if unit := _OME_TO_NGFF_LENGTH.get(getattr(pixels, field_name)):
                        axis["unit"] = unit
            yield axis

    def _build_datasets(
        self, ome: OME, series: int, resolution_count: int
    ) -> list[dict[str, Any]]:
        """Build datasets list with coordinate transforms for each resolution.

        Coordinate transformations include scale factors that account for:
        1. Physical pixel sizes (from OME metadata)
        2. Downsampling factors (ratio of resolution 0 to current resolution)

        For example, if resolution 0 is 4096x4096 @ 0.5 um/pixel and
        resolution 1 is 2048x2048, the downsampling factor is 2.0, so the
        effective scale becomes 0.5 * 2.0 = 1.0 um/pixel.
        """
        datasets: list[dict[str, Any]] = []

        # Get physical pixel sizes from OME metadata (resolution 0)
        pps = physical_pixel_sizes(ome, series)

        # Get reference dimensions from resolution 0
        meta_0 = self._biofile.core_metadata(series, 0)
        width_0 = meta_0.shape.x
        height_0 = meta_0.shape.y
        depth_0 = meta_0.shape.z

        for res in range(resolution_count):
            # Dataset path
            dataset: dict[str, Any] = {"path": str(res)}

            # Get dimensions for this resolution to calculate downsampling factor
            meta_r = self._biofile.core_metadata(series, res)
            width_r = meta_r.shape.x
            height_r = meta_r.shape.y
            depth_r = meta_r.shape.z

            # Calculate downsampling factors (how much smaller this resolution is)
            x_factor = width_0 / width_r if width_r > 0 else 1.0
            y_factor = height_0 / height_r if height_r > 0 else 1.0
            z_factor = depth_0 / depth_r if depth_r > 0 else 1.0

            # Build coordinate transforms (scale values)
            # Physical size * downsampling factor = effective pixel size
            # Only include scales for dimensions that are present (use same filter)
            # Use array store's dimension filter (single source of truth)
            store_0 = self._get_array_store(series, 0)
            dim_filter = store_0._dim_filter
            all_scales = [
                1.0,  # T
                1.0,  # C
                (pps.z * z_factor) if pps.z is not None else z_factor,  # Z
                (pps.y * y_factor) if pps.y is not None else y_factor,  # Y
                (pps.x * x_factor) if pps.x is not None else x_factor,  # X
            ]
            scale = [
                s for s, include in zip(all_scales, dim_filter, strict=False) if include
            ]

            dataset["coordinateTransformations"] = [{"type": "scale", "scale": scale}]
            datasets.append(dataset)

        return datasets

    def _get_array_store(self, series: int, resolution: int) -> BFArrayStore:
        """Get or create cached array store for a series/resolution.

        Uses integrated BioFormatsStore with RGB expansion and dimension
        squeezing flags. Much simpler than wrapping!
        """
        key = (series, resolution)
        if key not in self._array_stores:
            arr = self._biofile.as_array(series, resolution)
            # Single store with integrated transformations
            store = arr.zarr_store(
                tile_size=self._tile_size,
                rgb_as_channels=True,  # Interleave RGB into C (OME-Zarr convention)
                squeeze_singletons=True,  # Omit size-1 dims per NGFF
            )
            self._array_stores[key] = store
        return self._array_stores[key]

    # ------------------------------------------------------------------
    # Store ABC — required abstract methods
    # ------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
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

        if (parsed := ParsedPath.from_key(key)) is None:
            return None

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
            data = self._apply_byte_range(data, byte_range)

        return prototype.buffer.from_bytes(data)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the hierarchy."""
        try:
            if (parsed := ParsedPath.from_key(key)) is None:
                return False
        except ValueError:
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

    def close(self) -> None:
        """Close the store and all cached array stores."""
        self._is_open = False
        for store in self._array_stores.values():
            store.close()
        self._array_stores.clear()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._biofile.filename!r}, "
            f"series_count={len(self._biofile)})"
        )

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

    async def _copy_to(self, dest: Store) -> None:
        proto = default_buffer_prototype()
        async for key in self.list():
            buf = await self.get(key, prototype=proto)
            if buf is not None:
                await dest.set(key, buf)


# ==================================================================
# helper classes and functions


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


class ParsedPath(NamedTuple):
    """Structured information parsed from a zarr path."""

    level: PathLevel
    series: int | None = None
    resolution: int | None = None
    chunk_key: str | None = None

    @classmethod
    def from_key(cls, key: str) -> ParsedPath | None:
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
            return cls(PathLevel.ROOT)

        if key == "OME/zarr.json":
            return cls(PathLevel.OME_GROUP)

        if key == "OME/METADATA.ome.xml":
            return cls(PathLevel.OME_METADATA)

        parts = key.split("/")

        # Series/multiscales group: {series}/zarr.json
        if len(parts) == 2 and parts[1] == "zarr.json":
            return cls(PathLevel.MULTISCALES_GROUP, series=int(parts[0]))

        # Array metadata: {series}/{resolution}/zarr.json
        if len(parts) == 3 and parts[2] == "zarr.json":
            series = int(parts[0])
            resolution = int(parts[1])
            return cls(
                level=PathLevel.ARRAY_METADATA,
                series=series,
                resolution=resolution,
            )

        # Chunk data: {series}/{resolution}/c/...
        if len(parts) >= 4 and parts[2] == "c":
            series = int(parts[0])
            resolution = int(parts[1])
            # Reconstruct chunk key: "c/t/c/z/yi/xi" or "c/t/c/z/yi/xi/0"
            chunk_key = "/".join(parts[2:])
            return cls(
                level=PathLevel.CHUNK,
                series=series,
                resolution=resolution,
                chunk_key=chunk_key,
            )

        return None
