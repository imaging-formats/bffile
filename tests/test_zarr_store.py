"""Test zarr v3 store backed by Bio-Formats."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile

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
        store = bf.as_array().as_zarr()
        arr = zarr.open(store)
        assert isinstance(arr, zarr.Array)
        assert arr.shape == bf.as_array().shape
        assert arr.dtype == bf.as_array().dtype


def test_data_matches_read_plane(simple_file: Path) -> None:
    """Data read through zarr matches direct read_plane() calls."""
    with BioFile(simple_file) as bf:
        store = bf.as_array().as_zarr()
        zarr_arr = zarr.open_array(store)
        shape = bf.core_metadata().shape

        for t in range(shape.t):
            for c in range(shape.c):
                for z in range(shape.z):
                    expected = bf.read_plane(t=t, c=c, z=z, series=0, resolution=0)
                    np.testing.assert_array_equal(zarr_arr[t, c, z], expected)


def test_tiled_chunks(simple_file: Path) -> None:
    """Sub-plane tiling via tile_size produces correct data."""
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        store = arr.as_zarr(tile_size=(16, 16))
        zarr_arr = zarr.open_array(store)
        np.testing.assert_array_equal(arr, zarr_arr)


def test_multiseries(multiseries_file: Path) -> None:
    """Different series produce different zarr arrays."""
    with BioFile(multiseries_file) as bf:
        store0 = bf.as_array(series=0).as_zarr()
        store1 = bf.as_array(series=1).as_zarr()

        arr0 = zarr.open_array(store0)
        arr1 = zarr.open_array(store1)

        assert not np.allclose(arr0[0, 0], arr1[0, 0])


def test_rgb_image() -> None:
    """6D RGB arrays are handled correctly."""
    rgb_tiff = Path(__file__).parent / "data" / "s_1_t_1_c_2_z_1_RGB.tiff"
    with BioFile(rgb_tiff) as bf:
        arr = bf.as_array()
        assert arr.ndim == 6, f"Expected 6D RGB, got {arr.ndim}D"

        zarr_arr = zarr.open_array(arr.as_zarr())
        assert zarr_arr.shape == arr.shape == (1, 2, 1, 32, 32, 3)
        assert zarr_arr.ndim == arr.ndim == 6

        # Compare first plane
        expected = bf.read_plane(t=0, c=0, z=0)
        np.testing.assert_array_equal(zarr_arr[0, 0, 0], expected)


def test_read_only(simple_file: Path) -> None:
    """Store rejects write operations."""

    with BioFile(simple_file) as bf:
        store = bf.as_array().as_zarr()

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
        store1 = arr.as_zarr()
        store2 = arr.as_zarr()
        assert store1 == store2


def test_store_exists(simple_file: Path) -> None:
    """exists() returns True for zarr.json and chunk keys."""
    with BioFile(simple_file) as bf:
        store = bf.as_array().as_zarr()
        assert asyncio.run(store.exists("zarr.json"))
        assert asyncio.run(store.exists("c/0/0/0/0/0"))
        assert not asyncio.run(store.exists("nonexistent"))


def test_data_matches_multiseries(multiseries_file: Path) -> None:
    """Zarr data matches direct reads for multi-series file."""
    with BioFile(multiseries_file) as bf:
        for series_idx in range(min(len(bf), 2)):
            arr = bf.as_array(series=series_idx)
            zarr_arr = zarr.open_array(arr.as_zarr())

            # Compare a plane from each series
            expected = bf.read_plane(t=0, c=0, z=0, series=series_idx)
            actual = np.asarray(zarr_arr[0, 0, 0])
            np.testing.assert_array_equal(actual, expected)
