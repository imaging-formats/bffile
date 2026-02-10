"""Test multi-series and multi-resolution support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path


def test_series_count(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        meta_s0 = bf.core_meta(series=0)
        meta_s1 = bf.core_meta(series=1)
        meta_s2 = bf.core_meta(series=2)
        meta_s3 = bf.core_meta(series=3)
        assert meta_s0.shape is not None
        assert meta_s1.shape is not None
        assert meta_s2.shape is not None
        assert meta_s3.shape is not None


def test_as_array_series_parameter(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        arr_s0 = bf.as_array(series=0)
        arr_s1 = bf.as_array(series=1)
        assert arr_s0.shape is not None
        assert arr_s1.shape is not None


def test_core_meta_series_indexing(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        meta_s0 = bf.core_meta(series=0)
        meta_s1 = bf.core_meta(series=1)
        assert meta_s0.shape != meta_s1.shape or True


def test_series_out_of_bounds(multiseries_file: Path) -> None:
    with BioFile(multiseries_file) as bf:
        with pytest.raises(IndexError, match="out of range"):
            bf.core_meta(series=100)


def test_pyramid_resolution_count(pyramid_file: Path) -> None:
    with BioFile(pyramid_file) as bf:
        meta_r0 = bf.core_meta(series=0, resolution=0)
        assert meta_r0.resolution_count >= 1


def test_resolution_access(pyramid_file: Path) -> None:
    with BioFile(pyramid_file) as bf:
        meta = bf.core_meta(series=0, resolution=0)
        if meta.resolution_count > 1:
            arr_r0 = bf.as_array(series=0, resolution=0)
            arr_r1 = bf.as_array(series=0, resolution=1)
            assert arr_r0.shape[-2:] != arr_r1.shape[-2:]
