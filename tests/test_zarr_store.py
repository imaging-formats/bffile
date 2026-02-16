"""Test zarr v3 store backed by Bio-Formats."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile
from bffile._imread import imread, open_zarr_array

if TYPE_CHECKING:
    import zarr
    from zarr.core.buffer.core import default_buffer_prototype
else:
    try:
        import zarr
        from zarr.core.buffer.core import default_buffer_prototype
    except ImportError:
        pytest.skip(
            "Requires zarr v3 with buffer protocol support", allow_module_level=True
        )


def test_store_from_lazy_array(simple_file: Path) -> None:
    """as_zarr() on a LazyBioArray creates a usable store."""
    with BioFile(simple_file) as bf:
        arr = zarr.open(bf.as_array().to_zarr_store())
        lazy = bf.as_array()
    assert isinstance(arr, zarr.Array)
    assert arr.shape == lazy.shape
    assert arr.dtype == lazy.dtype


def test_data_matches_read_plane(any_file: Path) -> None:
    """Data read through zarr matches direct read_plane() calls."""
    if any_file.suffix.lower() in {".klb", ".svs"}:
        fmt = any_file.suffix
        pytest.xfail(f"Known issue with {fmt} files in zarr store")

    zarr_arr = open_zarr_array(any_file, series=0)
    assert isinstance(zarr_arr, zarr.Array)
    if zarr_arr.nbytes > 100 * 1024 * 1024:
        pytest.skip("File too large for in-memory comparison")

    all_data = imread(any_file)
    assert isinstance(all_data, np.ndarray)
    np.testing.assert_array_equal(zarr_arr, all_data)


def test_tiled_chunks(simple_file: Path) -> None:
    """Sub-plane tiling via tile_size produces correct data."""
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        store = arr.to_zarr_store(tile_size=(16, 16))
        zarr_arr = zarr.open_array(store)
        np.testing.assert_array_equal(arr, zarr_arr)


def test_multiseries(multiseries_file: Path) -> None:
    """Different series produce different zarr arrays."""
    with BioFile(multiseries_file) as bf:
        store0 = bf.as_array(series=0).to_zarr_store()
        store1 = bf.as_array(series=1).to_zarr_store()

        arr0 = zarr.open_array(store0)
        arr1 = zarr.open_array(store1)

        assert not np.allclose(arr0[0, 0], arr1[0, 0])


def test_rgb_image() -> None:
    """6D RGB arrays are handled correctly."""
    rgb_tiff = Path(__file__).parent / "data" / "s_1_t_1_c_2_z_1_RGB.tiff"
    with BioFile(rgb_tiff) as bf:
        arr = bf.as_array()
        assert arr.ndim == 6, f"Expected 6D RGB, got {arr.ndim}D"

        zarr_arr = zarr.open_array(arr.to_zarr_store())
        assert zarr_arr.shape == arr.shape == (1, 2, 1, 32, 32, 3)
        assert zarr_arr.ndim == arr.ndim == 6

        # Compare first plane
        expected = bf.read_plane(t=0, c=0, z=0)
        np.testing.assert_array_equal(zarr_arr[0, 0, 0], expected)


def test_read_only(simple_file: Path) -> None:
    """Store rejects write operations."""

    with BioFile(simple_file) as bf:
        store = bf.as_array().to_zarr_store()

        assert not store.supports_writes
        assert not store.supports_deletes

        proto = default_buffer_prototype()
        buf = proto.buffer.from_bytes(b"test")
        with pytest.raises(PermissionError, match="read-only"):
            asyncio.run(store.set("test_key", buf))
        with pytest.raises(PermissionError, match="read-only"):
            asyncio.run(store.delete("test_key"))


def test_store_equality(simple_file: Path) -> None:
    """Two stores from the same array are equal."""
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        store1 = arr.to_zarr_store()
        store2 = arr.to_zarr_store()
        assert store1 == store2


def test_store_exists(simple_file: Path) -> None:
    """exists() returns True for zarr.json and chunk keys."""
    with BioFile(simple_file) as bf:
        store = bf.as_array().to_zarr_store()
        assert asyncio.run(store.exists("zarr.json"))
        assert asyncio.run(store.exists("c/0/0/0/0/0"))
        assert not asyncio.run(store.exists("nonexistent"))


def test_data_matches_multiseries(multiseries_file: Path) -> None:
    """Zarr data matches direct reads for multi-series file."""
    with BioFile(multiseries_file) as bf:
        for series in bf:
            arr = series.as_array()
            zarr_arr = zarr.open_array(arr.to_zarr_store())

            # Compare a plane from each series
            expected = bf.read_plane(t=1, c=0, z=2, series=series.index)
            np.testing.assert_array_equal(zarr_arr[1, 0, 2], expected)


def test_invalid_chunk_keys(simple_file: Path) -> None:
    """Malformed chunk keys should return None/False and never raise."""
    with BioFile(simple_file) as bf:
        store = bf.as_array().to_zarr_store()
        proto = default_buffer_prototype()
        invalid_keys = [
            "c",
            "c/",
            "c/x/0/0/0/0",
            "c/0/0/0/0",
            "c/0/0/0/0/0/0",
            "c/999/0/0/0/0",
        ]

        for key in invalid_keys:
            assert asyncio.run(store.get(key, proto)) is None
            assert not asyncio.run(store.exists(key))
