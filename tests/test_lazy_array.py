"""Test LazyBioArray indexing and numpy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "indexing,squeezed_dims",
    [
        (np.s_[0, 0, 0], 3),  # Squeeze T, C, Z
        (np.s_[:], 0),  # No squeezing
        (np.s_[:2, 1:, 0:3], 0),  # No squeezing
        (np.s_[-1, -1, -1], 3),  # Squeeze T, C, Z
        (np.s_[..., 100:200], 0),  # No squeezing (ellipsis)
    ],
)
def test_indexing_patterns(
    opened_biofile: BioFile, indexing: tuple, squeezed_dims: int
) -> None:
    arr = opened_biofile.as_array()
    result = arr[indexing]
    expected_ndim = arr.ndim - squeezed_dims
    assert result.ndim == expected_ndim


def test_xy_subregion(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_meta()
    nt, nc, nz, ny, nx = meta.shape[:5]

    # Only test subregion if image is large enough
    if ny < 10 or nx < 10:
        pytest.skip("Image too small for subregion test")

    subregion = arr[:, :, :, 5:10, 5:10]
    expected_shape = (nt, nc, nz, 5, 5)
    if arr.is_rgb:
        expected_shape = expected_shape + (meta.shape[5],)
    assert subregion.shape == expected_shape


def test_mixed_indexing(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    result = arr[0, :, 0, :, :]
    # Squeezed T and Z, kept C, Y, X (and RGB if present)
    expected_ndim = arr.ndim - 2
    assert result.ndim == expected_ndim


def test_numpy_array_conversion(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    np_arr = np.array(arr)
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.shape == arr.shape
    assert np_arr.dtype == arr.dtype


def test_numpy_operations(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    max_proj = np.max(arr, axis=2)
    assert max_proj.ndim == arr.ndim - 1


def test_fancy_indexing_not_supported(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    with pytest.raises(NotImplementedError, match="fancy indexing"):
        arr[[0, 1, 2]]

    with pytest.raises(NotImplementedError, match="fancy indexing"):
        arr[np.array([0, 1, 2])]


def test_step_not_supported(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    with pytest.raises(NotImplementedError, match="step != 1"):
        arr[::2]


def test_empty_slice(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    result = arr[0:0, :, :]
    assert result.shape[0] == 0


def test_index_out_of_bounds(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    with pytest.raises(IndexError):
        arr[1000, 0, 0]


def test_shape_dtype_ndim_properties(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        assert isinstance(arr.shape, tuple)
        assert isinstance(arr.dtype, np.dtype)
        assert isinstance(arr.ndim, int)
        assert arr.ndim == len(arr.shape)


def test_repr(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        repr_str = repr(arr)
        assert "LazyBioArray" in repr_str
        assert "shape=" in repr_str
        assert "dtype=" in repr_str
