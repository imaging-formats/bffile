from pathlib import Path
from unittest.mock import patch

import pytest

from bffile import BioFile
from bffile._lazy_array import LazyBioArray

try:
    import xarray
except ImportError:
    pytest.skip("xarray is not installed", allow_module_level=True)


def test_to_xarray_basic(opened_biofile: BioFile) -> None:
    """Test basic to_xarray functionality."""
    lzarr = opened_biofile.as_array()
    xarr = opened_biofile.to_xarray()

    # ensure the underlying data is still a LazyBioArray
    assert isinstance(xarr, xarray.DataArray)
    assert isinstance(xarr.variable._data, LazyBioArray)
    assert tuple(xarr.dims) == lzarr.dims
    assert xarr.shape == lzarr.shape
    assert "ome_metadata" in xarr.attrs
    assert set(xarr.coords).issubset({"T", "C", "Z", "Y", "X", "S"})

    # now index into it, and ensure that LazyBioArray.__array__ is NOT called
    __array__ = LazyBioArray.__array__
    with patch.object(LazyBioArray, "__array__", autospec=True) as mock_array:
        mock_array.side_effect = __array__
        xarr_t0 = xarr.isel(T=0)
        assert "T" not in xarr_t0.dims
        assert isinstance(xarr_t0, xarray.DataArray)
        assert isinstance(xarr_t0.variable._data, LazyBioArray)
        mock_array.assert_not_called()
        _ = xarr_t0.data  # This should trigger __array__
        mock_array.assert_called_once_with(xarr_t0.variable._data)
        mock_array.reset_mock()

        xarr_t0c0 = xarr.isel(T=0, C=0)
        assert "C" not in xarr_t0c0.dims
        assert isinstance(xarr_t0c0, xarray.DataArray)
        assert isinstance(xarr_t0c0.variable._data, LazyBioArray)
        mock_array.assert_not_called()
        _ = xarr_t0c0.data  # This should trigger __array__
        mock_array.assert_called_with(xarr_t0c0.variable._data)


def test_to_xarray_rgb(rgb_file: Path) -> None:
    """Test to_xarray with RGB images."""
    with BioFile(rgb_file) as bf:
        xarr = bf.to_xarray()

        # Should work for RGB
        meta = bf.core_metadata(0, 0)
        assert xarr.shape == meta.shape.as_array_shape
        assert "S" in xarr.dims


def test_to_xarray_coords_from_ome(opened_biofile: BioFile) -> None:
    """Test that coordinates are properly derived from OME metadata."""
    xarr = opened_biofile.to_xarray()
    ome = opened_biofile.ome_metadata
    pixels = ome.images[0].pixels

    # Check channel names
    if pixels.channels:
        assert "C" in xarr.coords
        c_coords = xarr.coords["C"]
        assert len(c_coords) == len(pixels.channels)

    # Check physical coordinates are used when available
    if pixels.physical_size_x is not None:
        x_coords = xarr.coords["X"]
        assert len(x_coords) == pixels.size_x
        # First coord should be 0, second should be physical_size_x
        if pixels.size_x > 1:
            assert float(x_coords[1]) == pytest.approx(float(pixels.physical_size_x))
