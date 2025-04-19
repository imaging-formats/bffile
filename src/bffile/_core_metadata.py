from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CoreMetadata:
    size_x: int = 0
    size_y: int = 0
    size_z: int = 0
    size_c: int = 0
    size_t: int = 0
    thumb_size_x: int = 0
    thumb_size_y: int = 0
    pixel_type: int = 0
    bits_per_pixel: int = 0
    image_count: int = 0
    modulo_z: Any = None
    modulo_c: Any = None
    modulo_t: Any = None
    dimension_order: str = ""
    order_certain: bool = False
    rgb: bool = False
    little_endian: bool = False
    interleaved: bool = False
    indexed: bool = False
    false_color: bool = True
    metadata_complete: bool = False
    series_metadata: Dict[str, Any] = field(default_factory=dict)
    thumbnail: bool = False
    resolution_count: int = 1
