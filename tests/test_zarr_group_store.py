"""Tests for BioFormatsGroupStore - multi-series/multi-resolution zarr groups."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("zarr", reason="Requires zarr v3 with buffer protocol support")


def test_as_zarr_group_basic(simple_file: Path) -> None:
    """Test basic group store creation."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        assert store is not None
        assert store.supports_listing
        assert not store.supports_writes
        assert not store.supports_deletes


def test_root_metadata(simple_file: Path) -> None:
    """Test root group metadata structure (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        # Get root metadata
        import zarr

        data = asyncio.run(
            store.get("zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
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
        store = bf.as_zarr_group()
        import zarr

        # Test OME group metadata
        data = asyncio.run(
            store.get("OME/zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
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
        store = bf.as_zarr_group()
        import zarr

        # Get OME-XML
        proto = zarr.core.buffer.default_buffer_prototype()
        data = asyncio.run(store.get("OME/METADATA.ome.xml", proto))
        assert data is not None
        xml_str = data.to_bytes().decode()

        # Should be valid XML
        assert xml_str.startswith("<?xml") or xml_str.startswith("<OME")
        assert "OME" in xml_str


def test_series_metadata(simple_file: Path) -> None:
    """Test series/multiscales group metadata (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        import zarr

        # Test first series metadata (which IS the multiscales group)
        data = asyncio.run(
            store.get("0/zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
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
        store = bf.as_zarr_group()
        import zarr

        # Get multiscales metadata (series group IS multiscales in v0.5)
        data = asyncio.run(
            store.get("0/zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
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


def test_axes_structure(simple_file: Path) -> None:
    """Test axes metadata structure in detail."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        import zarr

        data = asyncio.run(
            store.get("0/zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
        metadata = json.loads(data.to_bytes())
        axes = metadata["attributes"]["ome"]["multiscales"][0]["axes"]

        # Arrays are always 5D (TCZYX) for non-RGB images
        # Always include all 5 axes regardless of size
        expected_axes = ["t", "c", "z", "y", "x"]

        axis_names = [ax["name"] for ax in axes]
        assert axis_names == expected_axes

        # Check axis types
        for axis in axes:
            if axis["name"] == "t":
                assert axis["type"] == "time"
            elif axis["name"] == "c":
                assert axis["type"] == "channel"
            else:
                assert axis["type"] == "space"


def test_array_metadata_delegation(simple_file: Path) -> None:
    """Test that array metadata is correctly delegated to BioFormatsStore."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        import zarr

        # Get array metadata
        data = asyncio.run(
            store.get("0/0/zarr.json", zarr.core.buffer.default_buffer_prototype())
        )
        assert data is not None
        metadata = json.loads(data.to_bytes())

        # Should be array metadata
        assert metadata["zarr_format"] == 3
        assert metadata["node_type"] == "array"
        assert "shape" in metadata
        assert "data_type" in metadata
        assert "chunk_grid" in metadata


def test_chunk_data_delegation(simple_file: Path) -> None:
    """Test that chunk data reads are correctly delegated."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        import zarr

        # Try to read first chunk
        # Chunk key format: {series}/{resolution}/c/{t}/{c}/{z}/{yi}/{xi}
        chunk_key = "0/0/c/0/0/0/0/0"

        # Check if chunk exists
        exists = asyncio.run(store.exists(chunk_key))

        if exists:
            data = asyncio.run(
                store.get(chunk_key, zarr.core.buffer.default_buffer_prototype())
            )
            assert data is not None
            # Should be raw bytes
            assert isinstance(data.to_bytes(), bytes)
            assert len(data.to_bytes()) > 0


def test_zarr_open_group(simple_file: Path) -> None:
    """Test opening the store with zarr.open_group."""
    import zarr

    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        group = zarr.open_group(store, mode="r")

        # Should be a zarr group
        assert isinstance(group, zarr.Group)

        # Should have OME subgroup
        assert "OME" in group.group_keys()

        # Should have series subgroups
        series_keys = [k for k in group.group_keys() if k.isdigit()]
        assert len(series_keys) == len(bf)


def test_zarr_access_array(simple_file: Path) -> None:
    """Test accessing arrays through the group."""
    import zarr

    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        group = zarr.open_group(store, mode="r")

        # Access first series, full resolution
        arr = group["0/0"]
        assert isinstance(arr, zarr.Array)

        # Check shape matches expected
        meta = bf[0].core_metadata()
        expected_shape = meta.shape.as_array_shape
        assert arr.shape == expected_shape

        # Read some data
        data = arr[0, 0, 0]
        assert data is not None


def test_list_keys(simple_file: Path) -> None:
    """Test listing all keys in the hierarchy (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()

        # List all keys
        async def _list_keys():
            return [k async for k in store.list()]

        keys = asyncio.run(_list_keys())

        # Should have root metadata
        assert "zarr.json" in keys

        # Should have OME keys
        assert "OME/zarr.json" in keys
        assert "OME/METADATA.ome.xml" in keys

        # Should have series keys (NGFF v0.5: no intermediate 0/ level)
        assert "0/zarr.json" in keys
        assert "0/0/zarr.json" in keys


def test_exists(simple_file: Path) -> None:
    """Test exists() method (NGFF v0.5)."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()

        # Valid keys
        assert asyncio.run(store.exists("zarr.json"))
        assert asyncio.run(store.exists("OME/zarr.json"))
        assert asyncio.run(store.exists("OME/METADATA.ome.xml"))
        assert asyncio.run(store.exists("0/zarr.json"))
        assert asyncio.run(store.exists("0/0/zarr.json"))

        # Invalid keys
        assert not asyncio.run(store.exists("nonexistent"))
        assert not asyncio.run(store.exists("999/zarr.json"))


def test_read_only(simple_file: Path) -> None:
    """Test that store is read-only."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()
        import zarr

        # Should raise on write attempts
        with pytest.raises(PermissionError):
            proto = zarr.core.buffer.default_buffer_prototype()
            asyncio.run(store.set("test", proto.buffer.from_bytes(b"data")))

        with pytest.raises(PermissionError):
            asyncio.run(store.delete("zarr.json"))


def test_multi_resolution(pyramid_file: Path) -> None:
    """Test multi-resolution support (NGFF v0.5)."""
    import zarr

    with BioFile(pyramid_file) as bf:
        # Check if file has multiple resolutions
        meta = bf[0].core_metadata()
        if meta.resolution_count <= 1:
            pytest.skip("File doesn't have multiple resolutions")

        store = bf.as_zarr_group()
        group = zarr.open_group(store, mode="r")

        # Check that we can access different resolutions
        arr0 = group["0/0"]  # Full resolution
        arr1 = group["0/1"]  # Downsampled

        # Resolution 1 should have smaller dimensions
        assert arr1.shape[-2:] < arr0.shape[-2:]


def test_tile_size_parameter(simple_file: Path) -> None:
    """Test that tile_size parameter is passed through to array stores."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group(tile_size=(256, 256))
        assert store is not None

        import zarr

        # Open and check chunk shape
        group = zarr.open_group(store, mode="r")
        arr = group["0/0"]

        # Chunk shape should use tile size
        # Note: first 3 dims are always (1, 1, 1) for T, C, Z
        assert arr.chunks[-2:] == (256, 256)


def test_path_router() -> None:
    """Test PathRouter parsing logic (NGFF v0.5)."""
    from bffile._zarr_group_store import PathLevel, PathRouter

    # Root
    parsed = PathRouter.parse("zarr.json")
    assert parsed.level == PathLevel.ROOT

    # OME group
    parsed = PathRouter.parse("OME/zarr.json")
    assert parsed.level == PathLevel.OME_GROUP

    # OME metadata
    parsed = PathRouter.parse("OME/METADATA.ome.xml")
    assert parsed.level == PathLevel.OME_METADATA

    # Series/multiscales group (NGFF v0.5: series IS multiscales)
    parsed = PathRouter.parse("0/zarr.json")
    assert parsed.level == PathLevel.MULTISCALES_GROUP
    assert parsed.series == 0

    # Array metadata
    parsed = PathRouter.parse("0/0/zarr.json")
    assert parsed.level == PathLevel.ARRAY_METADATA
    assert parsed.series == 0
    assert parsed.resolution == 0

    # Chunk
    parsed = PathRouter.parse("0/0/c/0/0/0/0/0")
    assert parsed.level == PathLevel.CHUNK
    assert parsed.series == 0
    assert parsed.resolution == 0
    assert parsed.chunk_key == "c/0/0/0/0/0"

    # Unknown
    parsed = PathRouter.parse("unknown/path")
    assert parsed.level == PathLevel.UNKNOWN


def test_open_zarr_with_series_none(simple_file: Path) -> None:
    """Test open_zarr with series=None returns a group."""

    # Import the function
    import zarr

    from bffile._imread import open_zarr

    result = open_zarr(simple_file, series=None)
    assert isinstance(result, zarr.Group)

    # Should have OME subgroup
    assert "OME" in result.group_keys()


def test_open_zarr_with_series_int(simple_file: Path) -> None:
    """Test open_zarr with series=int returns an array."""
    import zarr

    from bffile._imread import open_zarr

    result = open_zarr(simple_file, series=0)
    assert isinstance(result, zarr.Array)


def test_close(simple_file: Path) -> None:
    """Test closing the group store."""
    with BioFile(simple_file) as bf:
        store = bf.as_zarr_group()

        # Create some array stores by accessing data

        asyncio.run(store.exists("0/0/zarr.json"))

        # Close should clean up
        store.close()
        assert not store._is_open
        assert len(store._array_stores) == 0


def test_context_manager(simple_file: Path) -> None:
    """Test using group store as context manager."""
    with BioFile(simple_file) as bf:
        with bf.as_zarr_group() as store:
            assert store._is_open
        assert not store._is_open
