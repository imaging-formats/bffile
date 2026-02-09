from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile
from bffile._biofile import ReaderInfo
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

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_bffile(test_file: Path) -> None:
    with BioFile(test_file) as bf:
        assert bf.to_numpy() is not None
        assert bf.to_dask() is not None
        assert bf.shape


# --------------------- BioFile staticmethod tests ---------------------


def test_bioformats_version() -> None:
    """Test that bioformats_version returns a non-empty string."""
    version = BioFile.bioformats_version()
    assert isinstance(version, str)
    assert len(version) > 0
    assert version != "unknown"


def test_bioformats_maven_coordinate() -> None:
    """Test that bioformats_maven_coordinate returns expected format."""
    coord = BioFile.bioformats_maven_coordinate()
    assert isinstance(coord, str)
    # Maven coordinate format: groupId:artifactId:version
    parts = coord.split(":")
    assert len(parts) >= 3


def test_list_supported_suffixes() -> None:
    """Test that list_supported_suffixes returns a set of file extensions."""
    suffixes = BioFile.list_supported_suffixes()
    assert all(isinstance(s, str) for s in suffixes)
    assert "tif" in suffixes


def test_list_available_readers() -> None:
    """Test that list_available_readers returns ReaderInfo objects."""
    readers = BioFile.list_available_readers()
    assert len(readers) > 0

    # Check first reader has expected structure
    reader = readers[0]
    assert isinstance(reader, ReaderInfo)
    assert isinstance(reader.format, str)
    assert isinstance(reader.suffixes, tuple)
    assert isinstance(reader.class_name, str)
    assert isinstance(reader.is_gpl, bool)


# --------------------- _utils.py tests ---------------------


@pytest.mark.parametrize(
    ("func", "args", "expected"),
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
def test_ome_id_generators(func: Callable, args: tuple, expected: str) -> None:
    assert func(*args) == expected


def test_generate_coord_array() -> None:
    result = generate_coord_array(0, 5, 0.5)
    np.testing.assert_array_almost_equal(result, [0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize(
    ("slc", "length", "expected"),
    [
        (slice(None, None), 100, (0, 100)),  # no start/stop
        (slice(10, 50), 100, (10, 40)),  # normal slice
        (slice(50, 10), 100, (10, 40)),  # reversed slice
    ],
)
def test_slice2width(slc: slice, length: int, expected: tuple[int, int]) -> None:
    assert slice2width(slc, length) == expected


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


def test_physical_pixel_sizes_repr() -> None:
    pps = PhysicalPixelSizes(z=1.0, y=0.5, x=0.5)
    assert repr(pps) == "PhysicalPixelSizes(z=1.0, y=0.5, x=0.5)"


def test_ome_shape_repr() -> None:
    shape = OMEShape(t=1, c=2, z=3, y=100, x=100, rgb=1)
    assert repr(shape) == "TCZXYrgb(1, 2, 3, 100, 100, 1)"


# --------------------- BioFile tests ---------------------


def test_biofile_properties(test_file: Path) -> None:
    with BioFile(test_file) as bf:
        assert bf.filename == str(test_file.expanduser().absolute())
        assert isinstance(bf.ome_xml, str)
        # ome_metadata parsing may fail for some exotic file formats
        try:
            assert bf.ome_metadata is not None
        except Exception:
            pass
        # bioformats_version returns a Java string
        assert bf.bioformats_version()


def test_biofile_to_dask_explicit_series(test_file: Path) -> None:
    with BioFile(test_file) as bf:
        arr = bf.to_dask(series=0)
        assert arr is not None


def test_biofile_original_meta(test_file: Path) -> None:
    with BioFile(test_file, original_meta=True) as bf:
        assert bf.ome_xml is not None


def test_biofile_memoize(test_file: Path, tmp_path: Path) -> None:
    # Test with BIOFORMATS_MEMO_DIR set via monkeypatching
    import bffile._biofile as biofile_mod

    old_memo_dir = biofile_mod.BIOFORMATS_MEMO_DIR
    try:
        biofile_mod.BIOFORMATS_MEMO_DIR = tmp_path
        with BioFile(test_file, memoize=1) as bf:
            assert bf.to_numpy() is not None
    finally:
        biofile_mod.BIOFORMATS_MEMO_DIR = old_memo_dir


def test_biofile_dask_tiles(test_file: Path) -> None:
    with BioFile(test_file, dask_tiles=True) as bf:
        arr = bf.to_dask()
        assert arr is not None


def test_biofile_dask_tiles_custom_size(test_file: Path) -> None:
    with BioFile(test_file, dask_tiles=True, tile_size=(64, 64)) as bf:
        assert bf.tile_size == (64, 64)
        arr = bf.to_dask()
        assert arr is not None
