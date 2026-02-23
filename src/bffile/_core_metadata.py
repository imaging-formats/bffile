from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ._jimports import jimport

if TYPE_CHECKING:
    import loci.formats
    from typing_extensions import Self


class OMEShape(NamedTuple):
    """NamedTuple with OME metadata shape."""

    t: int
    c: int
    z: int
    y: int
    x: int
    rgb: int

    @property
    def as_array_shape(self) -> tuple[int, ...]:
        """Return 5-tuple `(T,C,Z,Y,X)` if `rgb==1`, else 6-tuple `(T,C,Z,Y,X,RGB)`."""
        if self.rgb > 1:
            return (self.t, self.c, self.z, self.y, self.x, self.rgb)
        return (self.t, self.c, self.z, self.y, self.x)

    def __repr__(self) -> str:
        return (
            f"(t={self.t}, c={self.c}, z={self.z}, "
            f"y={self.y}, x={self.x}, rgb={self.rgb})"
        )


def pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a loci.formats PixelType integer into a numpy dtype."""
    FormatTools = jimport("loci.formats.FormatTools")

    fmt2type: dict[int, str] = {
        FormatTools.INT8: "i1",
        FormatTools.UINT8: "u1",
        FormatTools.INT16: "i2",
        FormatTools.UINT16: "u2",
        FormatTools.INT32: "i4",
        FormatTools.UINT32: "u4",
        FormatTools.FLOAT: "f4",
        FormatTools.DOUBLE: "f8",
        FormatTools.BIT: "b1",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


@dataclass
class CoreMetadata:
    """Core metadata for a single series."""

    dtype: np.dtype
    shape: OMEShape
    rgb_count: int = 0
    thumb_size_x: int = 0
    thumb_size_y: int = 0
    bits_per_pixel: int = 0
    image_count: int = 0
    modulo_z: Any = None
    modulo_c: Any = None
    modulo_t: Any = None
    dimension_order: str = ""
    is_order_certain: bool = False
    is_rgb: bool = False
    is_little_endian: bool = False
    is_interleaved: bool = False
    is_indexed: bool = False
    is_false_color: bool = True
    is_metadata_complete: bool = False
    is_thumbnail_series: bool = False
    series_metadata: dict[str, Any] = field(default_factory=dict)
    resolution_count: int = 1

    @classmethod
    def from_java(cls, meta: loci.formats.CoreMetadata) -> Self:

        if size_zt := meta.sizeZ * meta.sizeT:
            eff_size_c = meta.imageCount // size_zt
        else:
            eff_size_c = 1

        if eff_size_c == 0:
            rgb_count = 1
        else:
            rgb_count = meta.sizeC // eff_size_c

        return cls(
            dtype=pixtype2dtype(meta.pixelType, meta.littleEndian),
            shape=OMEShape(
                x=meta.sizeX,
                y=meta.sizeY,
                z=meta.sizeZ,
                c=eff_size_c,
                t=meta.sizeT,
                rgb=rgb_count,
            ),
            rgb_count=rgb_count,
            thumb_size_x=meta.thumbSizeX,
            thumb_size_y=meta.thumbSizeY,
            bits_per_pixel=meta.bitsPerPixel,
            image_count=meta.imageCount,
            modulo_z=meta.moduloZ,
            modulo_c=meta.moduloC,
            modulo_t=meta.moduloT,
            dimension_order=str(meta.dimensionOrder),
            is_order_certain=meta.orderCertain,
            is_rgb=meta.rgb,
            is_little_endian=meta.littleEndian,
            is_interleaved=meta.interleaved,
            is_indexed=meta.indexed,
            is_false_color=meta.falseColor,
            is_metadata_complete=meta.metadataComplete,
            is_thumbnail_series=meta.thumbnail,
            series_metadata=dict(meta.seriesMetadata),
            resolution_count=meta.resolutionCount,
        )
