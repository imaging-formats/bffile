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
        darr = bf.to_dask(chunks=(1, 1, 1, -1, -1, -1))
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
    pytest.importorskip("dask")
    with BioFile(simple_file, dask_tiles=True, tile_size=(16, 16)) as bf:
        darr = bf.to_dask()
        assert darr.chunks is not None


def test_to_dask_tiles_with_auto_chunks(simple_file: Path) -> None:
    pytest.importorskip("dask")
    with BioFile(simple_file, dask_tiles=True) as bf:
        darr = bf.to_dask(chunks="auto")
        assert darr is not None
