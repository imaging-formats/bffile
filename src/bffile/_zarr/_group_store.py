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

from bffile._utils import physical_pixel_sizes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from ome_types import OME
    from pint import Quantity
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.storage import StoreLike

    from bffile._biofile import BioFile
    from bffile._core_metadata import CoreMetadata


# NGFF v0.5 specification unit whitelists
# https://ngff.openmicroscopy.org/v0.5/#axes-md
NGFF_LENGTH_UNITS = frozenset(
    [
        "angstrom",
        "attometer",
        "centimeter",
        "decimeter",
        "exameter",
        "femtometer",
        "foot",
        "gigameter",
        "hectometer",
        "inch",
        "kilometer",
        "megameter",
        "meter",
        "micrometer",
        "millimeter",
        "nanometer",
        "petameter",
        "picometer",
        "terameter",
        "yottameter",
        "zeptometer",
        "zettameter",
    ]
)
NGFF_TIME_UNITS = frozenset(
    [
        "attosecond",
        "centisecond",
        "day",
        "decisecond",
        "exasecond",
        "femtosecond",
        "gigasecond",
        "hectosecond",
        "hour",
        "kilosecond",
        "megasecond",
        "microsecond",
        "millisecond",
        "minute",
        "nanosecond",
        "picosecond",
        "second",
        "terasecond",
        "yottasecond",
        "zeptosecond",
        "zettasecond",
    ]
)


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


def _get_effective_shape(meta: CoreMetadata) -> tuple[int, ...]:
    """Get effective shape with RGB expanded into C dimension.

    Returns
    -------
    tuple[int, ...]
        Shape as (T, C_effective, Z, Y, X) where C_effective = C * RGB
    """
    return (
        meta.shape.t,
        meta.shape.c * meta.shape.rgb,  # Expand RGB into C
        meta.shape.z,
        meta.shape.y,
        meta.shape.x,
    )


def _get_dimension_filter(meta: CoreMetadata) -> list[bool]:
    """Determine which dimensions to include (True) or omit (False).

    **SINGLE SOURCE OF TRUTH** for dimension filtering across axes, scales,
    and array shapes.

    Omits dimensions with size 1, but always keeps Y and X (required by NGFF).
    This matches bioformats2raw's --compact behavior.

    Parameters
    ----------
    meta : CoreMetadata
        Core metadata with shape information

    Returns
    -------
    list[bool]
        Boolean mask [T, C, Z, Y, X] indicating which dimensions to include
    """
    t, c_eff, z, _y, _x = _get_effective_shape(meta)
    return [
        t > 1,  # T: only if size > 1
        c_eff > 1,  # C: only if size > 1 (RGB already expanded)
        z > 1,  # Z: only if size > 1
        True,  # Y: always include (NGFF requires 2 spatial dims)
        True,  # X: always include (NGFF requires 2 spatial dims)
    ]


def _extract_ngff_unit(
    quantity: Quantity | None, dimension_type: Literal["space", "time"]
) -> str | None:
    """Extract NGFF-compliant unit string from a pint Quantity.

    If the OME metadata contains a unit not in the whitelist, we omit the unit
    key from the axes metadata rather than writing an invalid value.
    """
    if quantity is None:
        return None

    # Extract unit string from pint Quantity
    unit_str = str(quantity.units)

    # Validate against NGFF whitelist
    if dimension_type == "space":
        if unit_str in NGFF_LENGTH_UNITS:
            return unit_str
    elif dimension_type == "time":
        if unit_str in NGFF_TIME_UNITS:
            return unit_str

    return None


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


class BFOmeZarrStore(Store):
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
        self._array_stores: dict[tuple[int, int], Store] = {}
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
        meta = self._biofile[series].core_metadata()
        ome = self._biofile.ome_metadata

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

        Omits singleton dimensions (size 1) except Y and X which are always
        included per NGFF requirements. Uses _get_dimension_filter() as the
        single source of truth.

        Units are extracted from OME-XML Quantity objects and validated against
        NGFF v0.5 whitelists.

        Note
        ----
        For RGB images, the C dimension is expanded to include RGB samples
        (e.g., C=2 with RGB=3 becomes C=6).
        """
        axes: list[dict[str, str]] = []
        pixels = ome.images[series].pixels
        dim_filter = _get_dimension_filter(meta)
        dim_names = ["t", "c", "z", "y", "x"]
        dim_types = ["time", "channel", "space", "space", "space"]

        for include, name, dim_type in zip(
            dim_filter, dim_names, dim_types, strict=False
        ):
            if not include:
                continue

            axis: dict[str, str] = {"name": name, "type": dim_type}

            # Add units for time and spatial dimensions
            if dim_type == "time":
                if unit := _extract_ngff_unit(pixels.time_increment_quantity, "time"):
                    axis["unit"] = unit
            elif dim_type == "space":
                quantity = getattr(pixels, f"physical_size_{name}_quantity")
                if unit := _extract_ngff_unit(quantity, "space"):
                    axis["unit"] = unit

            axes.append(axis)

        return axes

    def _build_datasets(
        self, meta: CoreMetadata, ome: OME, series: int
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

        for res in range(meta.resolution_count):
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
            dim_filter = _get_dimension_filter(meta_0)
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

    def _get_array_store(self, series: int, resolution: int) -> Store:
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

    def __enter__(self) -> BFOmeZarrStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._biofile.filename!r}, "
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
        raise PermissionError(f"{type(self).__name__} is read-only")

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
