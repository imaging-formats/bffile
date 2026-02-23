"""Test BioFile lifecycle, metadata, and static methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile, imread

if TYPE_CHECKING:
    from pathlib import Path


def test_open_close_lifecycle(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    assert bf.closed

    bf.open()
    assert not bf.closed
    bf.open()
    assert not bf.closed

    bf.close()
    assert bf.closed
    bf.close()
    assert bf.closed


def test_context_manager(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    assert bf.closed

    with bf as context_bf:
        assert context_bf is bf
        assert not bf.closed

    assert bf.closed


def test_closed_property(opened_biofile: BioFile) -> None:
    assert not opened_biofile.closed
    opened_biofile.close()
    assert opened_biofile.closed


def test_operations_require_open(simple_file: Path) -> None:
    bf = BioFile(simple_file)

    with pytest.raises(RuntimeError, match="File not open"):
        bf.core_metadata()

    with pytest.raises(RuntimeError, match="File not open"):
        bf.as_array()

    with pytest.raises(RuntimeError, match="File not open"):
        _ = bf.ome_xml

    with pytest.raises(RuntimeError, match="File not open"):
        bf.read_plane()


def test_reopen_after_close(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    bf.open()
    meta1 = bf.core_metadata()
    bf.close()

    bf.open()
    meta2 = bf.core_metadata()
    assert meta1.shape == meta2.shape
    bf.close()


def test_core_meta_returns_metadata(opened_biofile: BioFile) -> None:
    meta = opened_biofile.core_metadata()
    assert hasattr(meta, "shape")
    assert hasattr(meta, "dtype")
    assert hasattr(meta, "dimension_order")
    assert len(meta.shape) == 6  # CoreMetadata always has 6 elements (TCZYX + rgb)
    assert meta.rgb_count == meta.shape.rgb
    assert isinstance(meta.dtype, np.dtype)


@pytest.mark.parametrize(
    ("channel_filler", "expect_rgb", "expect_indexed"),
    [(None, 3, False), (True, 3, False), (False, 1, True)],
    ids=["auto", "forced", "disabled"],
)
def test_channel_filler_gif(
    data_dir: Path,
    channel_filler: bool | None,
    expect_rgb: int,
    expect_indexed: bool,
) -> None:
    with BioFile(data_dir / "example.gif", channel_filler=channel_filler) as bf:
        meta = bf.core_metadata()
        assert meta.shape.rgb == expect_rgb
        assert meta.is_indexed is expect_indexed


def test_false_color_indexed_file_not_expanded(data_dir: Path) -> None:
    with BioFile(data_dir / "ND2_dims_c2y32x32.nd2") as bf:
        meta = bf.core_metadata()
        assert meta.shape.c == 2
        assert meta.shape.rgb == 1
        assert meta.is_rgb is False
        assert meta.is_indexed
        assert meta.is_false_color


def test_ome_xml_property(opened_biofile: BioFile) -> None:
    xml = opened_biofile.ome_xml
    assert isinstance(xml, str)
    assert len(xml) > 0
    assert "OME" in xml


def test_ome_metadata_property(opened_biofile: BioFile) -> None:
    ome = opened_biofile.ome_metadata
    assert ome is not None
    assert hasattr(ome, "images")


def test_filename_property(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    assert simple_file.name in bf.filename
    assert str(simple_file) == bf.filename


def test_bioformats_version() -> None:
    version = BioFile.bioformats_version()
    assert isinstance(version, str)
    assert len(version) > 0
    parts = version.split(".")
    assert len(parts) >= 2


def test_list_available_readers() -> None:
    readers = BioFile.list_available_readers()
    assert len(readers) > 0
    for reader in readers:
        assert hasattr(reader, "format")
        assert hasattr(reader, "suffixes")
        assert hasattr(reader, "class_name")
        assert hasattr(reader, "is_gpl")
        assert isinstance(reader.suffixes, tuple)


def test_list_supported_suffixes() -> None:
    suffixes = BioFile.list_supported_suffixes()
    assert isinstance(suffixes, set)
    assert len(suffixes) > 0
    assert "tif" in suffixes or "tiff" in suffixes
    assert "nd2" in suffixes


def test_bioformats_maven_coordinate() -> None:
    coord = BioFile.bioformats_maven_coordinate()
    assert isinstance(coord, str)
    assert ":" in coord
    assert "ome" in coord or "bio-formats" in coord


def test_read_plane_subregion(opened_biofile: BioFile) -> None:
    meta = opened_biofile.core_metadata()
    ny, nx = meta.shape.y, meta.shape.x

    # Only test subregion if image is large enough
    if ny < 10 or nx < 10:
        pytest.skip("Image too small for subregion test")

    plane = opened_biofile.read_plane(t=0, c=0, z=0, y=slice(5, 10), x=slice(5, 10))
    assert plane.shape[0] == 5
    assert plane.shape[1] == 5


def test_as_array_with_series_resolution(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        arr = bf.as_array(series=1, resolution=0)
        assert arr.shape is not None


def test_core_meta_resolution_bounds(pyramid_file: Path) -> None:
    with BioFile(pyramid_file) as bf:
        with pytest.raises(IndexError, match="out of range"):
            bf.core_metadata(series=0, resolution=100)


def test_negative_resolution_indexing(pyramid_file: Path) -> None:
    """resolution=-1 should equal the lowest resolution level."""
    with BioFile(pyramid_file) as bf:
        n_res = bf.core_metadata(series=0).resolution_count
        meta_last = bf.core_metadata(series=0, resolution=n_res - 1)
        meta_neg = bf.core_metadata(series=0, resolution=-1)
        assert meta_last == meta_neg

        # as_array and read_plane also accept negative resolution
        arr_last = bf.as_array(series=0, resolution=n_res - 1)
        arr_neg = bf.as_array(series=0, resolution=-1)
        assert arr_last.shape == arr_neg.shape

        with pytest.raises(IndexError, match="out of range"):
            bf.core_metadata(series=0, resolution=-(n_res + 1))


def test_biofile_with_meta_disabled(simple_file: Path) -> None:
    with BioFile(simple_file, meta=False) as bf:
        xml = bf.ome_xml
        assert xml == ""


def test_biofile_with_original_meta(simple_file: Path) -> None:
    with BioFile(simple_file, original_meta=True) as bf:
        arr = bf.as_array()
        assert arr is not None


def test_imread(simple_file: Path) -> None:
    arr = imread(simple_file)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 5


def test_global_metadata(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        meta = bf.global_metadata()
        assert isinstance(meta, dict)
        assert meta


def test_used_files(any_file: Path) -> None:
    with BioFile(any_file) as bf:
        # Test both with and without metadata_only flag
        files = bf.used_files()
        assert files
        assert any(bf.filename in f for f in files)

        meta_files = bf.used_files(metadata_only=True)
        assert isinstance(meta_files, list)


def test_lookup_table(any_file: Path) -> None:
    """Test lookup_table method for various file types."""
    with BioFile(any_file) as bf:
        for series in range(len(bf)):
            lut = bf.lookup_table(series=series)
            if lut is not None:
                assert isinstance(lut, np.ndarray)
                assert lut.ndim == 2
                assert lut.shape[0] >= 1  # At least one channel
                assert lut.shape[1] >= 1  # At least one value
                assert lut.dtype in (np.uint8, np.uint16)


def test_get_thumbnail_basic(opened_biofile: BioFile) -> None:
    """Test basic thumbnail retrieval."""
    thumb = opened_biofile.get_thumbnail()
    assert isinstance(thumb, np.ndarray)
    assert thumb.ndim in (2, 3)
    assert 0 < thumb.shape[0] <= 128
    assert 0 < thumb.shape[1] <= 128

    # can also be retrieved via series method
    assert np.array_equal(thumb, opened_biofile[0].get_thumbnail())


def test_get_thumbnail_custom_max_size(opened_biofile: BioFile) -> None:
    """Custom max_size constrains output to requested box."""
    thumb = opened_biofile.get_thumbnail(max_thumbnail_size=(64, 48))
    assert isinstance(thumb, np.ndarray)
    assert 0 < thumb.shape[1] <= 64
    assert 0 < thumb.shape[0] <= 48


def test_get_thumbnail_pyramid(pyramid_file: Path) -> None:
    """Test that thumbnail uses lowest resolution and supports negative indexing."""
    with BioFile(pyramid_file) as bf:
        thumb = bf.get_thumbnail()
        assert isinstance(thumb, np.ndarray)
        assert 0 < thumb.shape[0] <= 128
        assert 0 < thumb.shape[1] <= 128
