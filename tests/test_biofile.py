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
        bf.core_meta()

    with pytest.raises(RuntimeError, match="File not open"):
        bf.as_array()

    with pytest.raises(RuntimeError, match="File not open"):
        _ = bf.ome_xml

    with pytest.raises(RuntimeError, match="File not open"):
        bf.read_plane()


def test_reopen_after_close(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    bf.open()
    meta1 = bf.core_meta()
    bf.close()

    bf.open()
    meta2 = bf.core_meta()
    assert meta1.shape == meta2.shape
    bf.close()


def test_core_meta_returns_metadata(opened_biofile: BioFile) -> None:
    meta = opened_biofile.core_meta()
    assert hasattr(meta, "shape")
    assert hasattr(meta, "dtype")
    assert hasattr(meta, "dimension_order")
    assert len(meta.shape) == 6  # CoreMetadata always has 6 elements (TCZYX + rgb)
    assert isinstance(meta.dtype, np.dtype)


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
    meta = opened_biofile.core_meta()
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
            bf.core_meta(series=0, resolution=100)


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
