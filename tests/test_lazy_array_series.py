"""Test LazyBioArray multi-series independence."""

from __future__ import annotations

import numpy as np
import pytest

from bffile import BioFile


def test_lazy_array_multi_series_independence():
    """Test that multiple LazyBioArray instances with different series are independent."""
    test_file = "tests/data/s_3_t_1_c_3_z_5.czi"

    with BioFile(test_file) as bf:
        series_count = bf._java_reader.getSeriesCount()

        # Skip if not multi-series
        if series_count < 2:
            pytest.skip("File does not have multiple series")

        # Get ground truth from to_numpy()
        truth_s0 = bf.to_numpy(series=0)
        truth_s1 = bf.to_numpy(series=1)

        # Ensure they're actually different
        assert not np.array_equal(
            truth_s0, truth_s1
        ), "Test requires series with different data"

        # Create lazy arrays for both series
        arr0 = bf.as_array(series=0)
        arr1 = bf.as_array(series=1)

        # Read from series 0
        lazy_s0_first = arr0[0, 0, 0]

        # Read from series 1 (this used to corrupt arr0's reads)
        lazy_s1 = arr1[0, 0, 0]

        # Read from series 0 again - should still get series 0 data
        lazy_s0_second = arr0[0, 0, 0]

        # Verify all reads are correct
        assert np.array_equal(
            lazy_s0_first, truth_s0[0, 0, 0]
        ), "First read from arr0 should match series 0"

        assert np.array_equal(
            lazy_s1, truth_s1[0, 0, 0]
        ), "Read from arr1 should match series 1"

        assert np.array_equal(
            lazy_s0_second, truth_s0[0, 0, 0]
        ), "Second read from arr0 should still match series 0"

        # Verify consistency
        assert np.array_equal(
            lazy_s0_first, lazy_s0_second
        ), "Multiple reads from arr0 should be consistent"


def test_lazy_array_interleaved_reads():
    """Test interleaved reads from multiple series."""
    test_file = "tests/data/s_3_t_1_c_3_z_5.czi"

    with BioFile(test_file) as bf:
        series_count = bf._java_reader.getSeriesCount()

        if series_count < 2:
            pytest.skip("File does not have multiple series")

        # Create arrays
        arr0 = bf.as_array(series=0)
        arr1 = bf.as_array(series=1)

        # Get ground truth
        truth_s0 = bf.to_numpy(series=0)
        truth_s1 = bf.to_numpy(series=1)

        # Interleave reads
        results = []
        for i in range(3):
            # Read from series 0, then series 1, then series 0 again
            r0 = arr0[0, i % arr0.shape[1], 0]
            r1 = arr1[0, i % arr1.shape[1], 0]
            r0_again = arr0[0, i % arr0.shape[1], 0]

            results.append((r0, r1, r0_again))

        # Verify all reads
        for i, (r0, r1, r0_again) in enumerate(results):
            c = i % arr0.shape[1]
            assert np.array_equal(
                r0, truth_s0[0, c, 0]
            ), f"Iteration {i}: arr0 should match series 0"
            assert np.array_equal(
                r1, truth_s1[0, c, 0]
            ), f"Iteration {i}: arr1 should match series 1"
            assert np.array_equal(r0, r0_again), f"Iteration {i}: arr0 reads inconsistent"


def test_lazy_array_numpy_protocol_preserves_series():
    """Test that np.array() conversion uses correct series."""
    test_file = "tests/data/s_3_t_1_c_3_z_5.czi"

    with BioFile(test_file) as bf:
        series_count = bf._java_reader.getSeriesCount()

        if series_count < 2:
            pytest.skip("File does not have multiple series")

        # Create lazy arrays
        arr0 = bf.as_array(series=0)
        arr1 = bf.as_array(series=1)

        # Convert to numpy
        numpy_arr0 = np.array(arr0)
        numpy_arr1 = np.array(arr1)

        # Get ground truth
        truth_s0 = bf.to_numpy(series=0)
        truth_s1 = bf.to_numpy(series=1)

        # Verify
        assert np.array_equal(
            numpy_arr0, truth_s0
        ), "np.array(arr0) should match series 0"
        assert np.array_equal(
            numpy_arr1, truth_s1
        ), "np.array(arr1) should match series 1"
