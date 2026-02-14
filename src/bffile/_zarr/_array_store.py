"""Read-only zarr v3 store backed by Bio-Formats."""

from __future__ import annotations

import json
import math
from itertools import compress, product
from typing import TYPE_CHECKING, Any

import numpy as np

from bffile._zarr._base_store import ReadOnlyStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

    from bffile._biofile import BioFile


class BFArrayStore(ReadOnlyStore):
    """Read-only zarr v3 store that virtualizes a Bio-Formats series/resolution.

    Parameters
    ----------
    biofile : BioFile
        BioFile instance to read from. Caller manages the lifecycle.
    series : int
        Series index to virtualize
    resolution : int, optional
        Resolution level (0 = full resolution), by default 0
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
    Via LazyBioArray convenience method:

    >>> with BioFile("image.nd2") as bf:
    ...     store = bf.as_array().zarr_store()
    ...     arr = zarr.open(store, mode="r")
    ...     data = arr[0, 0, 0]

    Direct construction:

    >>> with BioFile("image.nd2") as bf:
    ...     store = BFArrayStore(bf, series=0, resolution=0)
    ...     arr = zarr.open(store, mode="r")
    """

    def __init__(
        self,
        biofile: BioFile,
        series: int,
        resolution: int = 0,
        /,
        *,
        tile_size: tuple[int, int] | None = None,
        rgb_as_channels: bool = False,
        squeeze_singletons: bool = False,
    ) -> None:
        super().__init__(read_only=True)
        self._biofile = biofile
        self._series = series
        self._resolution = resolution
        self._meta = biofile.core_metadata(series, resolution)
        self._tile_size = tile_size
        self._rgb_as_channels = rgb_as_channels
        self._array_metadata_bytes: bytes | None = None  # built lazily
        self._chunk_keys: set[str] | None = None
        self._is_open = True

        self._dim_filter = self._compute_dimension_filter(squeeze_singletons)
        self._effective_shape = self._compute_effective_shape()

    def _compute_dimension_filter(self, squeeze_singletons: bool) -> list[bool]:
        """Return mask indicating which dims to include.

        Returns a 5- or 6-element list of booleans corresponding to
        (T, C, Z, Y, X[, RGB]) dimensions.
        """
        shape = self._meta.shape
        allow_6d = shape.rgb > 1 and not self._rgb_as_channels
        dim_filter = [True] * (6 if allow_6d else 5)
        if squeeze_singletons:
            nc = shape.c * shape.rgb if self._rgb_as_channels else shape.c
            dim_filter[:3] = [shape.t > 1, nc > 1, shape.z > 1]
        return dim_filter

    def _compute_effective_shape(self) -> tuple[int, ...]:
        """Compute effective shape with RGB handling and dimension squeezing."""
        shp = list(self._meta.shape)  # TCZYXr, len=6
        if self._rgb_as_channels:  # Interleave RGB samples as separate C channels
            shp[1] *= shp.pop()
        elif shp[-1] <= 1:
            shp.pop()
        return tuple(compress(shp, self._dim_filter))

    def dimension_names(self) -> Iterator[str]:
        """Return active dimension names.

        Returns
        -------
        Iterator[str]
            Subset of "tczyxr" for non-squeezed dimensions.
        """
        yield from compress("tczyxr", self._dim_filter)

    # ------------------------------------------------------------------
    # Metadata & chunk key helpers
    # ------------------------------------------------------------------

    def _array_metadata(self) -> bytes:
        """Build and cache zarr.json metadata as bytes."""
        if self._array_metadata_bytes is not None:
            return self._array_metadata_bytes  # pragma: no cover

        meta = self._meta
        shape = meta.shape

        # Determine chunks based on tile size and dimension filtering
        ty, tx = self._tile_size if self._tile_size else (shape.y, shape.x)
        chunk_shape = [1, 1, 1, ty, tx]
        if not self._rgb_as_channels and shape.rgb > 1:
            chunk_shape.append(shape.rgb)
        chunk_shape = list(compress(chunk_shape, self._dim_filter))

        metadata: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(self._effective_shape),
            "data_type": np.dtype(meta.dtype).name,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunk_shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [
                {
                    "name": "bytes",
                    "configuration": {
                        "endian": "little" if meta.is_little_endian else "big"
                    },
                },
            ],
        }

        self._array_metadata_bytes = json.dumps(metadata).encode()
        return self._array_metadata_bytes

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
        # Add RGB range if present (always range(1) - never chunk along RGB)
        ranges = [t_range, c_range, z_range, yi_range, xi_range]
        if not self._rgb_as_channels and shape.rgb > 1:
            ranges.append(range(1))  # Single chunk for RGB dimension

        # Apply dimension filter to generate squeezed keys
        for indices in product(*compress(ranges, self._dim_filter)):
            keys.add(f"c/{'/'.join(str(i) for i in indices)}")

        self._chunk_keys = keys
        return keys

    def _parse_chunk_key(self, key: str) -> tuple[int, int, int, int, int]:
        """Parse squeezed chunk key to (t, c, z, yi, xi) coordinates.

        Note: Returns 5D TCZYX coordinates. If key includes RGB coordinate
        (for 6D arrays), it's ignored since RGB is always 0.
        """
        parts = key.split("/")
        squeezed_indices = (int(p) for p in parts[1:])  # Skip "c" prefix

        # Map squeezed indices back to full dimensions (5D or 6D)
        # We always return 5D (TCZYX), ignoring RGB coordinate if present
        has_rgb_dim = not self._rgb_as_channels and self._meta.shape.rgb > 1
        ndims = 6 if has_rgb_dim else 5

        full_indices = [0] * ndims  # [t, c, z, yi, xi] or [t, c, z, yi, xi, rgb]
        kept_dims = (i for i, keep in enumerate(self._dim_filter) if keep)
        for dim_i, val in zip(kept_dims, squeezed_indices, strict=False):
            full_indices[dim_i] = val

        # Return only TCZYX (first 5), dropping RGB coordinate if present
        return tuple(full_indices[:5])  # type: ignore[return-value]

    def _read_chunk(self, key: str) -> bytes:
        """Read a chunk by its key and return raw bytes."""
        t, c_eff, z, yi, xi = self._parse_chunk_key(key)

        # Map effective C back to base (c, rgb_sample) if RGB interleaved as channels
        if self._rgb_as_channels:
            c_base, rgb_sample = divmod(c_eff, self._meta.shape.rgb)
        else:
            c_base = c_eff
            rgb_sample = None

        shape = self._meta.shape

        # Calculate Y, X slices for this chunk
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
            c=c_base,
            z=z,
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
            series=self._series,
            resolution=self._resolution,
        )

        # Extract single RGB channel if interleaving RGB as channels
        if rgb_sample is not None and shape.rgb > 1:
            plane = plane[..., rgb_sample]

        # Pad edge chunks to full tile size (zarr expects full chunk shape)
        if self._tile_size is not None:
            ty, tx = self._tile_size
            actual_h, actual_w = plane.shape[:2]
            if actual_h < ty or actual_w < tx:
                is_2d = rgb_sample is not None or shape.rgb == 1
                pad_shape = (ty, tx) if is_2d else (ty, tx, shape.rgb)
                padded = np.zeros(pad_shape, dtype=self._meta.dtype)
                if is_2d:
                    padded[:actual_h, :actual_w] = plane
                else:
                    padded[:actual_h, :actual_w, :] = plane
                plane = padded

        return np.ascontiguousarray(plane).tobytes()

    # ------------------------------------------------------------------
    # Store ABC — required abstract methods
    # ------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return NotImplemented  # pragma: no cover
        return (
            self._biofile.filename == value._biofile.filename
            and self._series == value._series
            and self._resolution == value._resolution
            and self._tile_size == value._tile_size
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if key == "zarr.json":
            data = self._array_metadata()
        elif key in self._get_chunk_keys():
            with self._biofile.ensure_open():
                data = self._read_chunk(key)
        else:
            return None

        if byte_range is not None:
            data = self._apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def exists(self, key: str) -> bool:
        return key == "zarr.json" or key in self._get_chunk_keys()

    async def list(self) -> AsyncIterator[str]:
        yield "zarr.json"
        for key in self._get_chunk_keys():
            yield key

    def close(self) -> None:
        """Close the store (and owned BioFile, if any)."""
        self._is_open = False  # pragma: no cover

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._biofile.filename!r}, "
            f"series={self._series}, shape={self._effective_shape})"
        )
