"""Test dask integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path


def test_to_dask_returns_array(simple_file: Path) -> None:
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        darr = bf.to_dask()
        assert hasattr(darr, "compute")
        assert hasattr(darr, "shape")
        assert hasattr(darr, "dtype")


def test_to_dask_custom_chunks(simple_file: Path) -> None:
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        darr = bf.to_dask(chunks=(1, 1, 1, -1, -1))
        assert darr.chunks is not None


def test_to_dask_compute(simple_file: Path) -> None:
    dask = pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        darr = bf.to_dask()
        with dask.config.set(scheduler="synchronous"):
            result = darr.compute()
        assert result.shape == darr.shape


def test_to_dask_import_error(
    simple_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "dask.array", None)
    with BioFile(simple_file) as bf:
        with pytest.raises(ImportError, match="Dask is required"):
            bf.to_dask()


def test_to_dask_with_tiles(simple_file: Path) -> None:
    """Test tile-based chunking with explicit tile size."""
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        darr = bf.to_dask(tile_size=(16, 16))
        assert darr.chunks is not None


def test_to_dask_tiles_auto(simple_file: Path) -> None:
    """Test tile-based chunking with auto-computed tile size."""
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        darr = bf.to_dask(tile_size="auto")
        assert darr.chunks is not None


def test_to_dask_tile_size_chunks_mutually_exclusive(simple_file: Path) -> None:
    """Test that tile_size and chunks cannot be used together."""
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        with pytest.raises(ValueError, match="mutually exclusive"):
            bf.to_dask(chunks=(1, 1, 1, -1, -1), tile_size=(512, 512))


def test_to_dask_tile_size_validation(simple_file: Path) -> None:
    """Test that tile_size is validated."""
    pytest.importorskip("dask")
    with BioFile(simple_file) as bf:
        with pytest.raises(ValueError, match="tile_size must be"):
            bf.to_dask(tile_size=(512,))  # type: ignore[arg-type]
