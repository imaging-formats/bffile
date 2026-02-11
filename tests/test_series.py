"""Tests for Series proxy and BioFile sequence protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile, Series

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# BioFile sequence protocol
# ---------------------------------------------------------------------------


def test_len(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        assert len(bf) == bf.series_count() == 4


def test_len_single_series(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        assert len(bf) == 1


def test_len_requires_open(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    with pytest.raises(RuntimeError, match="not open"):
        len(bf)


def test_getitem(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        s = bf[0]
        assert isinstance(s, Series)
        assert s.index == 0


def test_getitem_negative(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        last = bf[-1]
        assert isinstance(last, Series)
        assert last.index == len(bf) - 1


def test_getitem_out_of_bounds(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        with pytest.raises(IndexError):
            bf[100]
        with pytest.raises(IndexError):
            bf[-100]


def test_getitem_type_error(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        with pytest.raises(TypeError):
            bf["foo"]  # type: ignore[index]


def test_iter(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        series_list = list(bf)
        assert len(series_list) == len(bf)
        assert all(isinstance(s, Series) for s in series_list)


def test_biofile_repr_open(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        r = repr(bf)
        assert "BioFile" in r
        assert "4 series" in r


def test_biofile_repr_closed(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    r = repr(bf)
    assert "BioFile" in r
    assert "closed" in r


# ---------------------------------------------------------------------------
# Series properties
# ---------------------------------------------------------------------------


def test_series_properties(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        assert bf[0].is_rgb is False
        assert bf[2].index == 2
        for i in range(len(bf)):
            s = bf[i]
            full_shape = bf.core_metadata(series=i).shape
            nt, nc, nz, ny, nx = full_shape[:5]
            nrgb = full_shape[5] if len(full_shape) == 6 else 1
            expected = (nt, nc, nz, ny, nx, nrgb) if nrgb > 1 else (nt, nc, nz, ny, nx)
            assert s.shape == expected
            assert bf[i].dtype == bf.core_metadata(series=i).dtype
            assert bf[i].is_thumbnail == bf.core_metadata(series=i).is_thumbnail_series
            assert bf[i].resolution_count == bf.core_metadata(series=i).resolution_count


# ---------------------------------------------------------------------------
# Series methods
# ---------------------------------------------------------------------------


def test_series_core_meta(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        assert bf[1].core_metadata() == bf.core_metadata(series=1)


def test_series_as_array(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        for i in range(len(bf)):
            arr = bf[i].as_array()
            expected = bf.as_array(series=i)
            assert arr.shape == expected.shape
            assert arr.dtype == expected.dtype


def test_series_to_dask(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        darr = bf[0].to_dask()
        assert darr.shape == bf[0].shape


def test_series_read_plane(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        for i in range(len(bf)):
            plane = bf[i].read_plane()
            expected = bf.read_plane(series=i)
            np.testing.assert_array_equal(plane, expected)


def test_series_repr(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        r = repr(bf[0])
        assert "Series" in r
        assert "index=" in r
        assert "shape=" in r
        assert "dtype=" in r


def test_getitem_slice(multiseries_file: Path) -> None:
    """Test BioFile.__getitem__ with slices."""
    with BioFile(multiseries_file) as bf:
        # Test basic slice
        series_list = bf[0:2]
        assert isinstance(series_list, list)
        assert len(series_list) == 2
        assert all(isinstance(s, Series) for s in series_list)
        assert series_list[0].index == 0
        assert series_list[1].index == 1

        # Test slice with step
        stepped = bf[0::2]
        assert len(stepped) == 2
        assert stepped[0].index == 0
        assert stepped[1].index == 2

        # Test full slice
        all_series = bf[:]
        assert len(all_series) == len(bf)


def test_series_used_files(multiseries_file: Path) -> None:
    """Test Series.used_files method."""
    with BioFile(multiseries_file) as bf:
        files = bf[0].used_files()
        assert isinstance(files, list)
        assert len(files) >= 1

        meta_files = bf[0].used_files(metadata_only=True)
        assert isinstance(meta_files, list)
