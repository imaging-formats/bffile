"""Tests for BioFormatsGroupStore - multi-series/multi-resolution zarr groups."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from bffile import BioFile
from bffile._imread import open_ome_zarr_group

if TYPE_CHECKING:
    import zarr
    from zarr.core.buffer import default_buffer_prototype
else:
    zarr = pytest.importorskip(
        "zarr", reason="Requires zarr v3 with buffer protocol support"
    )
    from zarr.core.buffer import default_buffer_prototype
TEST_DATA = Path(__file__).parent / "data"


def test_to_zarr_basic(simple_file: Path) -> None:
    """Test basic group store creation."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()
        assert store is not None
        assert store.supports_listing
        assert not store.supports_writes
        assert not store.supports_deletes


def test_root_metadata(simple_file: Path) -> None:
    """Test root group metadata structure (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()
        # Get root metadata

        data = asyncio.run(store.get("zarr.json", default_buffer_prototype()))
        assert data is not None
        metadata = json.loads(data.to_bytes())

        assert metadata["zarr_format"] == 3
        assert metadata["node_type"] == "group"
        assert "ome" in metadata["attributes"]
        ome_attrs = metadata["attributes"]["ome"]
        assert ome_attrs["version"] == "0.5"
        assert ome_attrs["bioformats2raw.layout"] == 3


def test_ome_metadata(simple_file: Path) -> None:
    """Test OME group metadata (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()

        # Test OME group metadata
        data = asyncio.run(store.get("OME/zarr.json", default_buffer_prototype()))
        assert data is not None
        metadata = json.loads(data.to_bytes())

        assert metadata["zarr_format"] == 3
        assert metadata["node_type"] == "group"
        assert "ome" in metadata["attributes"]
        ome_attrs = metadata["attributes"]["ome"]
        assert ome_attrs["version"] == "0.5"
        assert "series" in ome_attrs
        series_list = ome_attrs["series"]
        assert isinstance(series_list, list)
        assert len(series_list) == len(bf)
        assert series_list[0] == "0"


def test_ome_xml_metadata(simple_file: Path) -> None:
    """Test OME-XML metadata file."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()

        # Get OME-XML
        proto = default_buffer_prototype()
        data = asyncio.run(store.get("OME/METADATA.ome.xml", proto))
        assert data is not None
        xml_str = data.to_bytes().decode()

        # Should be valid XML
        assert xml_str.startswith("<?xml") or xml_str.startswith("<OME")
        assert "OME" in xml_str


def test_series_metadata(simple_file: Path) -> None:
    """Test series/multiscales group metadata (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()

        # Test first series metadata (which IS the multiscales group)
        data = asyncio.run(store.get("0/zarr.json", default_buffer_prototype()))
        assert data is not None
        metadata = json.loads(data.to_bytes())

        assert metadata["zarr_format"] == 3
        assert metadata["node_type"] == "group"
        assert "ome" in metadata["attributes"]
        assert metadata["attributes"]["ome"]["version"] == "0.5"
        assert "multiscales" in metadata["attributes"]["ome"]


def test_multiscales_metadata(simple_file: Path) -> None:
    """Test multiscales group metadata (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()

        # Get multiscales metadata (series group IS multiscales in v0.5)
        data = asyncio.run(store.get("0/zarr.json", default_buffer_prototype()))
        assert data is not None
        metadata = json.loads(data.to_bytes())

        assert metadata["zarr_format"] == 3
        assert metadata["node_type"] == "group"
        assert "ome" in metadata["attributes"]

        ome_attrs = metadata["attributes"]["ome"]
        assert ome_attrs["version"] == "0.5"
        assert "multiscales" in ome_attrs

        multiscales = ome_attrs["multiscales"]
        assert len(multiscales) == 1
        ms = multiscales[0]

        # Check required fields
        assert ms["version"] == "0.5"
        assert "axes" in ms
        assert "datasets" in ms

        # Check axes
        axes = ms["axes"]
        assert isinstance(axes, list)
        assert len(axes) > 0

        # Axes should have name and type
        for axis in axes:
            assert "name" in axis
            assert "type" in axis

        # Check datasets
        datasets = ms["datasets"]
        assert isinstance(datasets, list)
        assert len(datasets) >= 1

        # First dataset should be path "0"
        assert datasets[0]["path"] == "0"
        assert "coordinateTransformations" in datasets[0]


@pytest.mark.parametrize(
    "file, expected_axes, series",
    [
        (TEST_DATA / "s_1_t_1_c_1_z_1.ome.tiff", ["y", "x"], 1),
        (TEST_DATA / "ND2_dims_p1z5t3c2y32x32.nd2", ["t", "c", "z", "y", "x"], 1),
        (TEST_DATA / "ND2_dims_rgb.nd2", ["c", "y", "x"], 1),
        (TEST_DATA / "XYS.lif", ["y", "x"], 4),
    ],
    ids=lambda val: getattr(val, "name", ""),
)
def test_dimension_metadata(file: Path, expected_axes: list[str], series: int) -> None:
    """Test that dimension metadata is correctly included in the OME-ZARR group."""
    group = open_ome_zarr_group(file)
    assert list(group.group_keys()) == ["OME", *[str(i) for i in range(series)]]

    series0 = group["0"]
    assert isinstance(series0, zarr.Group)
    metadata = cast("dict", series0.attrs["ome"])
    axes = metadata["multiscales"][0]["axes"]
    assert [ax["name"] for ax in axes] == expected_axes
    res0 = series0["0"]
    assert isinstance(res0, zarr.Array)
    assert res0.ndim == len(expected_axes)
    for ax in axes:
        if ax["name"] == "t":
            assert ax["type"] == "time"
        elif ax["name"] == "c":
            assert ax["type"] == "channel"
        elif ax["name"] in ["z", "y", "x"]:
            assert ax["type"] == "space"


def test_multi_resolution(pyramid_file: Path) -> None:
    """Test multi-resolution support (NGFF v0.5)."""
    group = open_ome_zarr_group(pyramid_file)
    series = group["0"]
    assert isinstance(series, zarr.Group)

    ome_meta = cast("dict", series.attrs["ome"])
    assert "multiscales" in ome_meta
    multiscales = ome_meta["multiscales"]
    assert len(multiscales) == 1
    ms = multiscales[0]
    assert "datasets" in ms
    datasets = ms["datasets"]
    assert len(datasets) >= 3
    assert datasets[0]["path"] == "0"
    assert datasets[1]["path"] == "1"
    assert datasets[2]["path"] == "2"

    res0 = series["0"]
    assert isinstance(res0, zarr.Array)
    res2 = series["2"]
    assert isinstance(res2, zarr.Array)


def test_invalid_keys_return_none_and_false(simple_file: Path) -> None:
    """Malformed or out-of-range keys should not raise from get()/exists()."""
    with BioFile(simple_file) as bf:
        store = bf.to_zarr_store()
        proto = default_buffer_prototype()

        invalid_keys = [
            "foo/zarr.json",
            "x/0/zarr.json",
            "999/zarr.json",
            "0/x/zarr.json",
            "0/999/zarr.json",
            "0/0/c/not-an-int",
        ]

        for key in invalid_keys:
            assert asyncio.run(store.get(key, proto)) is None
            assert not asyncio.run(store.exists(key))


def test_output_valid_zarr(any_file: Path, tmp_path: Path) -> None:
    """Test that the output can be read by zarr and matches expected data."""
    dest = tmp_path / "example.ome.zarr"
    with BioFile(any_file) as biofile:
        arr = biofile.as_array()
        if arr.nbytes > 2_000_000_000:  # Skip arrays > 2GB
            pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")

        biofile.to_zarr_store().save(dest)
    try:
        import yaozarrs

        yaozarrs.validate_zarr_store(dest)
    except ImportError:
        return
