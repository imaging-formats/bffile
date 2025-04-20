from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np


class OMEShape(NamedTuple):
    """NamedTuple with OME metadata shape."""

    t: int
    c: int
    z: int
    y: int
    x: int
    rgb: int

    def __repr__(self) -> str:
        return f"TCZXYrgb({self.t}, {self.c}, {self.z}, {self.y}, {self.x}, {self.rgb})"


@dataclass
class CoreMetadata:
    dtype: np.dtype
    size_x: int = 0
    size_y: int = 0
    size_z: int = 0
    size_c: int = 0
    size_t: int = 0
    rgb_count: int = 0
    series_count: int = 0
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

    @property
    def shape(self) -> OMEShape:
        """Return the shape of the image data."""
        return OMEShape(
            self.size_t,
            self.size_c,
            self.size_z,
            self.size_y,
            self.size_x,
            self.rgb_count,
        )
