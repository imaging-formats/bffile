"""Test edge cases, validation, and warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path


def test_rgb_ndim_six(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        meta = bf.core_metadata()
        if meta.shape[5] > 1:
            assert arr.ndim == 6
        else:
            assert arr.ndim == 5


def test_non_rgb_ndim_five(rgb_file: Path) -> None:
    with BioFile(rgb_file) as bf:
        arr = bf.as_array()
        meta = bf.core_metadata()
        if meta.shape[5] <= 1:
            assert arr.ndim == 5
        else:
            assert arr.ndim == 6


def test_rgb_interleaved_layout(rgb_file: Path) -> None:
    with BioFile(rgb_file) as bf:
        arr = bf.as_array()
        plane = arr[0, 0, 0]
        if arr.ndim == 6:
            assert plane.shape[-1] in (3, 4)


def test_memoization_parameter_accepted(simple_file: Path) -> None:
    bf = BioFile(simple_file, memoize=1000)
    with bf:
        arr = bf.as_array()
        assert arr is not None


def test_empty_xy_slice(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    result = arr[:, :, :, 0:0, :]
    assert result.shape[3] == 0


def test_read_plane_out_of_bounds(opened_biofile: BioFile) -> None:
    meta = opened_biofile.core_metadata()
    with pytest.raises(Exception):  # noqa: B017
        opened_biofile.read_plane(t=meta.shape[0] + 100)


def test_finalizer_cleanup_on_gc(simple_file: Path) -> None:
    import gc
    import weakref

    bf = BioFile(simple_file)
    bf.open()
    weak_ref = weakref.ref(bf)

    assert weak_ref() is not None
    del bf
    gc.collect()
    assert weak_ref() is None


def test_as_array_requires_open(simple_file: Path) -> None:
    bf = BioFile(simple_file)
    with pytest.raises(RuntimeError):
        bf.as_array()


def test_multiple_lazy_arrays(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        arr1 = bf.as_array()
        arr2 = bf.as_array()
        assert arr1.shape == arr2.shape
