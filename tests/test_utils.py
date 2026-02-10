"""Test utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from bffile._utils import (
    OMEShape,
    PhysicalPixelSizes,
    _chunk_by_tile_size,
    axis_id_to_slice,
    generate_coord_array,
    generate_ome_channel_id,
    generate_ome_detector_id,
    generate_ome_image_id,
    generate_ome_instrument_id,
    slice2width,
)


@pytest.mark.parametrize(
    "slc,length,expected",
    [
        (slice(None, None), 100, (0, 100)),
        (slice(10, 50), 100, (10, 40)),
        (slice(50, 10), 100, (10, 40)),  # Reversed slice
        (slice(0, 0), 100, (0, 0)),  # Empty slice
    ],
)
def test_slice2width(slc: slice, length: int, expected: tuple[int, int]) -> None:
    assert slice2width(slc, length) == expected


def test_ome_shape_repr() -> None:
    shape = OMEShape(t=1, c=2, z=3, y=100, x=100, rgb=1)
    assert repr(shape) == "TCZXYrgb(1, 2, 3, 100, 100, 1)"
    assert shape.t == 1
    assert shape.c == 2
    assert shape.z == 3
    assert shape.y == 100
    assert shape.x == 100
    assert shape.rgb == 1


def test_ome_shape_indexing() -> None:
    shape = OMEShape(t=10, c=3, z=5, y=512, x=512, rgb=1)
    assert shape[0] == 10  # t
    assert shape[1] == 3  # c
    assert shape[2] == 5  # z
    assert shape[3] == 512  # y
    assert shape[4] == 512  # x
    assert shape[5] == 1  # rgb
    assert len(shape) == 6


@pytest.mark.parametrize(
    "func,args,expected",
    [
        (generate_ome_image_id, ("0",), "Image:0"),
        (generate_ome_image_id, (1,), "Image:1"),
        (generate_ome_channel_id, ("Image:0", "1"), "Channel:0:1"),
        (generate_ome_channel_id, ("Image:2", 3), "Channel:2:3"),
        (generate_ome_instrument_id, ("0",), "Instrument:0"),
        (generate_ome_instrument_id, (1,), "Instrument:1"),
        (generate_ome_detector_id, ("cam",), "Detector:cam"),
        (generate_ome_detector_id, (0,), "Detector:0"),
    ],
)
def test_ome_id_generators(func, args: tuple, expected: str) -> None:
    assert func(*args) == expected


@pytest.mark.parametrize(
    ("n_px", "tile_length", "expected"),
    [
        (1024, 256, (256, 256, 256, 256)),  # exact division
        (1000, 256, (256, 256, 256, 232)),  # partial last tile
    ],
)
def test_chunk_by_tile_size(
    n_px: int, tile_length: int, expected: tuple[int, ...]
) -> None:
    assert _chunk_by_tile_size(n_px, tile_length) == expected


def test_generate_coord_array() -> None:
    result = generate_coord_array(0, 5, 0.5)
    np.testing.assert_array_almost_equal(result, [0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize(
    ("axis_id", "tile_length", "n_px", "expected"),
    [
        (0, 256, 1024, slice(0, 256)),  # first tile
        (3, 256, 1024, slice(768, 1024)),  # last exact tile
        (4, 256, 1100, slice(1024, 1100)),  # edge tile (partial)
    ],
)
def test_axis_id_to_slice(
    axis_id: int, tile_length: int, n_px: int, expected: slice
) -> None:
    assert axis_id_to_slice(axis_id, tile_length, n_px) == expected


def test_physical_pixel_sizes_repr() -> None:
    """Test PhysicalPixelSizes repr (currently commented out in _utils.py).

    This test documents the expected behavior if the class is uncommented.
    """

    pps = PhysicalPixelSizes(z=1.0, y=0.5, x=0.5)
    assert repr(pps) == "PhysicalPixelSizes(z=1.0, y=0.5, x=0.5)"
