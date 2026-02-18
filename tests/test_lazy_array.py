"""Test LazyBioArray indexing and numpy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile, LazyBioArray

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
    if arr.nbytes > 1_000_000_000:  # Skip arrays > 1GB
        pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")
    result = arr[indexing]
    expected_ndim = arr.ndim - squeezed_dims
    assert result.ndim == expected_ndim


def test_xy_subregion(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_metadata()
    nt, nc, nz, ny, nx = meta.shape[:5]

    # Only test subregion if image is large enough
    if ny < 10 or nx < 10:
        pytest.skip("Image too small for subregion test")

    subregion = arr[:, :, :, 5:10, 5:10]
    expected_shape = (nt, nc, nz, 5, 5)
    if arr.is_rgb:
        expected_shape = (*expected_shape, meta.shape[5])
    assert subregion.shape == expected_shape


def test_mixed_indexing(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    if arr.nbytes > 1_000_000_000:  # Skip arrays > 1GB
        pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")

    result = arr[0, :, 0, :, :]
    # Squeezed T and Z, kept C, Y, X (and RGB if present)
    expected_ndim = arr.ndim - 2
    assert result.ndim == expected_ndim


def test_numpy_array_conversion(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    if arr.nbytes > 1_000_000_000:  # Skip arrays > 1GB
        pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")

    np_arr = np.array(arr)
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.shape == arr.shape
    assert np_arr.dtype == arr.dtype


def test_numpy_operations(opened_biofile: BioFile) -> None:
    arr = opened_biofile.as_array()
    if arr.nbytes > 1_000_000_000:  # Skip arrays > 1GB
        pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")
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


def test_shape_size_dtype_ndim_properties(simple_file: Path) -> None:
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        assert isinstance(arr.shape, tuple)
        assert isinstance(arr.size, int)
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


def test_conditional_dimension_slicing(opened_biofile: BioFile) -> None:
    """Test slicing along dimensions that exist."""
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_metadata()
    nt, nc, nz = meta.shape.t, meta.shape.c, meta.shape.z

    # Only test multi-timepoint slicing if nt > 1
    if nt > 1:
        result = arr[:2, 0, 0]
        assert result.shape[0] == 2

    # Only test multi-channel slicing if nc > 1
    if nc > 1:
        result = arr[0, :, 0]
        assert result.shape[0] == nc

    # Only test multi-z slicing if nz > 1
    if nz > 1:
        result = arr[0, 0, :]
        assert result.shape[0] == nz


def test_partial_key_indexing(opened_biofile: BioFile) -> None:
    """Test indexing with fewer dimensions than array has."""
    arr = opened_biofile.as_array()
    if arr.nbytes > 1_000_000_000:  # Skip arrays > 1GB
        pytest.skip(f"Array too large ({arr.nbytes / 1e9:.2f} GB)")

    # Should implicitly add full slices for missing dimensions
    result = arr[0]
    expected_ndim = arr.ndim - 1
    assert result.ndim == expected_ndim


def test_dimension_squeezing(opened_biofile: BioFile) -> None:
    """Test that integer indexing properly squeezes dimensions."""
    arr = opened_biofile.as_array()

    if arr.is_rgb:
        # 6D → 3D by fixing T, C, Z (leaves Y, X, RGB)
        plane = arr[0, 0, 0]
        assert plane.ndim == 3
        assert plane.shape[-1] in (3, 4)  # RGB or RGBA
    else:
        # 5D → 2D by fixing T, C, Z (leaves Y, X)
        plane = arr[0, 0, 0]
        assert plane.ndim == 2


def test_lazy_vs_numpy_single_plane(opened_biofile: BioFile) -> None:
    """Verify lazy array returns same data as direct numpy conversion."""
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_metadata()

    # Use lower resolution for pyramid files to avoid 2GB limit
    if meta.resolution_count > 1:
        arr = opened_biofile.as_array(resolution=1)

    # Skip if plane would be too large (>100MB to keep test fast)
    frame_size = meta.shape.y * meta.shape.x * meta.shape.rgb
    plane_size_mb = (frame_size * meta.dtype.itemsize) / (1024 * 1024)
    if plane_size_mb > 100:
        pytest.skip("Plane too large for fast correctness test")

    # Get ground truth via full materialization
    numpy_data = np.asarray(arr)

    # Test single plane
    lazy_plane = np.asarray(arr[0, 0, 0])
    numpy_plane = numpy_data[0, 0, 0]

    assert np.array_equal(lazy_plane, numpy_plane)


def test_lazy_vs_numpy_subregion(opened_biofile: BioFile) -> None:
    """Verify subregion reads match numpy."""
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_metadata()
    ny, nx = meta.shape.y, meta.shape.x

    # Skip if image too small or too large
    if ny < 20 or nx < 20:
        pytest.skip("Image too small for subregion test")

    plane_size_mb = (ny * nx * meta.shape.rgb * meta.dtype.itemsize) / (1024 * 1024)
    if plane_size_mb > 100:
        pytest.skip("Plane too large for fast correctness test")

    # Get ground truth
    numpy_data = np.asarray(arr)

    # Test subregion (center 10x10 pixels)
    y_mid, x_mid = ny // 2, nx // 2
    y_slice = slice(y_mid - 5, y_mid + 5)
    x_slice = slice(x_mid - 5, x_mid + 5)

    lazy_roi = np.asarray(arr[0, 0, 0, y_slice, x_slice])
    numpy_roi = numpy_data[0, 0, 0, y_slice, x_slice]

    assert np.array_equal(lazy_roi, numpy_roi)


def test_multi_series_independence(multiseries_file) -> None:
    """Critical: Verify interleaved reads from multiple series don't corrupt data."""
    with BioFile(multiseries_file) as bf:
        # Get ground truth for both series
        truth_s0 = np.asarray(bf.as_array(series=0))
        truth_s1 = np.asarray(bf.as_array(series=1))

        arr0 = bf.as_array(series=0)
        arr1 = bf.as_array(series=1)

        lazy_s0_first = np.asarray(arr0[0, 0, 0])
        lazy_s1 = np.asarray(arr1[0, 0, 0])
        lazy_s0_second = np.asarray(arr0[0, 0, 0])

        assert np.array_equal(lazy_s0_first, truth_s0[0, 0, 0])
        assert np.array_equal(lazy_s1, truth_s1[0, 0, 0])
        assert np.array_equal(lazy_s0_second, truth_s0[0, 0, 0])
        assert np.array_equal(lazy_s0_first, lazy_s0_second)


def test_tile_height_calculation(simple_file: Path) -> None:
    """Test that tile height calculation respects Java limit."""
    with BioFile(simple_file) as bf:
        meta = bf.core_metadata(0, 0)

        tile_height = bf._calculate_tile_height(meta, meta.shape.x)
        tile_bytes = tile_height * meta.shape.x * meta.dtype.itemsize * meta.shape.rgb
        assert tile_bytes <= 2**31 - 1
        assert tile_height >= 1


def test_tiled_vs_direct_read_consistency(
    simple_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that tiled reads produce same results as direct reads.

    Monkeypatch the MAX_JAVA_ARRAY_SIZE to a smaller value to force multiple tiles for
    small sample.
    """
    import bffile._biofile as _biofile

    monkeypatch.setattr(_biofile, "MAX_JAVA_ARRAY_SIZE", 2**16)
    with BioFile(simple_file) as bf:
        reader = bf._ensure_java_reader()
        meta = bf.core_metadata(0, 0)

        height, width = meta.shape.y, meta.shape.x
        direct = bf._read_plane_direct(reader, meta, 0, 0, 0, 0, 0, height, width)
        tiled = bf._read_plane_tiled(reader, meta, 0, 0, 0, 0, 0, height, width)
        np.testing.assert_array_equal(direct, tiled)


def test_tiled_read_subregion(simple_file: Path) -> None:
    """Test that tiled reads work correctly for subregions."""
    with BioFile(simple_file) as bf:
        reader = bf._ensure_java_reader()
        meta = bf.core_metadata(0, 0)

        # Skip if image too small
        if meta.shape.y < 20 or meta.shape.x < 20:
            pytest.skip("Image too small for subregion test")

        y_start, x_start = 5, 10
        height, width = 10, 10
        direct = bf._read_plane_direct(
            reader, meta, 0, 0, 0, y_start, x_start, height, width
        )
        tiled = bf._read_plane_tiled(
            reader, meta, 0, 0, 0, y_start, x_start, height, width
        )
        np.testing.assert_array_equal(direct, tiled)


@pytest.mark.parametrize(
    ("ops", "direct", "min_t", "min_c", "min_z"),
    [
        # Composition: each operation applies to first effective dimension
        ((slice(0, 10), slice(2, 4)), np.s_[2:4], 10, 1, 1),  # ar[0:10][2:4] == [2:4]
        ((slice(0, 5), -1), np.s_[4], 5, 1, 1),  # arr[0:5][-1] == arr[4]
        ((0, slice(0, 2)), np.s_[0, 0:2], 1, 2, 1),  # arr[0][0:2] == arr[0, 0:2]
        ((1, 0, 1), np.s_[1, 0, 1], 2, 1, 2),  # arr[1][0][1] == arr[1, 0, 1]
        ((slice(0, 2), 1), np.s_[1], 2, 1, 1),  # arr[0:2][1] == arr[1]
    ],
)
def test_lazy_view_compositions(
    opened_biofile: BioFile,
    ops: tuple[slice | int, ...],
    direct: tuple[slice | int, ...] | slice | int,
    min_t: int,
    min_c: int,
    min_z: int,
) -> None:
    """Composed lazy indexing matches direct indexing for common patterns."""
    arr = opened_biofile.as_array()
    meta = opened_biofile.core_metadata()

    if arr.nbytes > 100_000_000:
        pytest.skip("Array too large")

    if meta.shape.t < min_t or meta.shape.c < min_c or meta.shape.z < min_z:
        pytest.skip("Array does not satisfy composition requirements")

    view = arr
    for index in ops:
        view = view[index]
        if isinstance(view, LazyBioArray):
            assert view.ndim <= arr.ndim

    composed = np.asarray(view)
    expected = np.asarray(arr[direct])
    np.testing.assert_array_equal(composed, expected)


def test_rgb_index_composition_uses_parent_bounds(rgb_file: Path) -> None:
    """Nested RGB indexing should compose with the parent RGB slice."""
    with BioFile(rgb_file) as bf:
        arr = bf.as_array()
        meta = bf.core_metadata()

        assert arr.ndim == 6
        assert arr.is_rgb and meta.shape.rgb >= 3

        plane = arr[0, 0, 0]
        assert plane.ndim == 3
        composed = plane[..., 1:3][..., 0]
        assert composed.ndim == 2

        assert isinstance(composed, LazyBioArray)
        assert composed._bounds_tczyxs[5] == slice(1, 2)
        assert composed._squeezed_tczyxs[5]

        materialized = np.asarray(composed)
        expected = np.asarray(arr[0, 0, 0, :, :, 1])
        np.testing.assert_array_equal(materialized, expected)


def test_scalar_index_all_dimensions(simple_file: Path, rgb_file: Path) -> None:
    """Regression: indexing all dimensions to a scalar must not raise IndexError."""
    with BioFile(simple_file) as bf:
        arr = bf.as_array()
        scalar = np.asarray(arr[0, 0, 0, 0, 0])
        assert scalar.ndim == 0
        _ = scalar.item()  # should not raise
        full = np.asarray(arr)
        assert scalar[()] == full[0, 0, 0, 0, 0]

    with BioFile(rgb_file) as bf:
        arr = bf.as_array()
        assert arr.ndim == 6  # t, c, z, y, x, rgb
        scalar = np.asarray(arr[0, 0, 0, 0, 0, 0])
        assert scalar.ndim == 0
        _ = scalar.item()  # should not raise
        full = np.asarray(arr)
        assert scalar[()] == full[0, 0, 0, 0, 0, 0]


def test_dimension_squeezing_composition(opened_biofile: BioFile) -> None:
    """Successive integer indexing squeezes dimensions as expected."""
    arr = opened_biofile.as_array()

    view = arr[0]
    assert isinstance(view, LazyBioArray)
    assert view.ndim == arr.ndim - 1

    view = view[0]
    assert isinstance(view, LazyBioArray)
    assert view.ndim == arr.ndim - 2

    view = view[0]
    assert isinstance(view, LazyBioArray)
    expected_ndim = 3 if arr.is_rgb else 2
    assert view.ndim == expected_ndim
