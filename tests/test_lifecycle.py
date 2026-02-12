"""Tests for BioFile lifecycle: open / close / destroy state transitions."""

from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bffile import BioFile

if TYPE_CHECKING:
    from pathlib import Path


def _assert_uninitialized(bf: BioFile) -> None:
    """Assert BioFile is in UNINITIALIZED state."""
    assert bf.closed
    assert bf._java_reader is None
    assert bf._core_meta_list is None
    assert bf._suspended is False
    assert bf._finalizer is None


def _assert_open(bf: BioFile) -> None:
    """Assert BioFile is in OPEN state and can read data."""
    assert not bf.closed
    assert bf._java_reader is not None
    assert bf._core_meta_list is not None
    assert bf._suspended is False
    plane = bf.read_plane()
    assert isinstance(plane, np.ndarray)


def _assert_suspended(bf: BioFile) -> None:
    """Assert BioFile is SUSPENDED: metadata accessible, reads blocked."""
    assert bf.closed
    assert bf._java_reader is not None  # reader preserved
    assert bf._core_meta_list is not None  # metadata preserved
    assert bf._suspended is True
    # Metadata works
    assert len(bf) > 0
    bf.core_metadata()
    # Data reads blocked
    with pytest.raises(RuntimeError, match="not open"):
        bf.read_plane()
    with pytest.raises(RuntimeError, match="not open"):
        bf._ensure_java_reader()
    with pytest.raises(RuntimeError, match="not open"):
        bf.as_array()


# ---------------------------------------------------------------------------
# UNINITIALIZED state
# ---------------------------------------------------------------------------


def test_uninitialized(simple_file: Path) -> None:
    """BioFile starts UNINITIALIZED; all operations fail; close/destroy no-op."""
    bf = BioFile(simple_file)
    _assert_uninitialized(bf)

    for method in (
        bf._ensure_java_reader,
        bf.read_plane,
        bf.as_array,
        bf.core_metadata,
    ):
        with pytest.raises(RuntimeError, match="not open"):
            method()  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="not open"):
        len(bf)

    # close and destroy are safe no-ops
    bf.close()
    _assert_uninitialized(bf)
    bf.destroy()
    _assert_uninitialized(bf)


# ---------------------------------------------------------------------------
# UNINITIALIZED -> OPEN -> SUSPENDED -> OPEN -> ... -> destroy
# ---------------------------------------------------------------------------


def test_full_lifecycle(simple_file: Path) -> None:
    """Walk through every transition: open, close, reopen, destroy, reopen."""
    bf = BioFile(simple_file)

    # UNINITIALIZED -> OPEN
    result = bf.open()
    assert result is bf  # returns self for chaining
    _assert_open(bf)
    reader_first = bf._java_reader
    meta_before = bf.core_metadata()

    # open() again is a no-op — same reader
    bf.open()
    assert bf._java_reader is reader_first

    # OPEN -> SUSPENDED
    bf.close()
    _assert_suspended(bf)
    assert bf._java_reader is reader_first  # same Java object
    assert bf.core_metadata() == meta_before  # metadata unchanged

    # close() again is idempotent
    bf.close()
    _assert_suspended(bf)

    # SUSPENDED -> OPEN (fast path — same reader)
    bf.open()
    _assert_open(bf)
    assert bf._java_reader is reader_first
    assert bf.core_metadata() == meta_before

    # Multiple close/open cycles reuse the same reader
    for _ in range(3):
        bf.close()
        _assert_suspended(bf)
        bf.open()
        _assert_open(bf)
        assert bf._java_reader is reader_first

    # destroy -> UNINITIALIZED
    bf.destroy()
    _assert_uninitialized(bf)

    # destroy again is idempotent
    bf.destroy()
    _assert_uninitialized(bf)

    # Can reopen from UNINITIALIZED (slow path — new reader)
    bf.open()
    _assert_open(bf)
    assert bf._java_reader is not reader_first
    bf.destroy()


def test_destroy_from_open_and_suspended(simple_file: Path) -> None:
    """destroy() works from both OPEN and SUSPENDED states."""
    # From OPEN
    bf = BioFile(simple_file)
    bf.open()
    bf.destroy()
    _assert_uninitialized(bf)

    # From SUSPENDED
    bf.open()
    bf.close()
    bf.destroy()
    _assert_uninitialized(bf)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(simple_file: Path) -> None:
    """with block opens on enter, destroys on exit, supports re-entry."""
    bf = BioFile(simple_file)

    # First context: open -> destroy
    with bf:
        _assert_open(bf)
        reader_first = bf._java_reader
    _assert_uninitialized(bf)

    # Re-enter: full re-init with new reader
    with bf:
        _assert_open(bf)
        assert bf._java_reader is not reader_first
        meta = bf.core_metadata()

    # Second re-entry: metadata matches
    with bf:
        meta2 = bf.core_metadata()
        assert meta.shape == meta2.shape
        assert meta.dtype == meta2.dtype


def test_close_open_inside_context(simple_file: Path) -> None:
    """close()/open() inside a with block uses the fast path."""
    with BioFile(simple_file) as bf:
        reader_original = bf._java_reader

        bf.close()
        _assert_suspended(bf)
        assert bf._java_reader is reader_original

        bf.open()
        _assert_open(bf)
        assert bf._java_reader is reader_original  # fast path

    # __exit__ destroys
    _assert_uninitialized(bf)


# ---------------------------------------------------------------------------
# GC finalizer
# ---------------------------------------------------------------------------


def test_gc_finalizer_from_open(simple_file: Path) -> None:
    """del bf from OPEN triggers the GC finalizer, fully closing the reader."""
    bf = BioFile(simple_file)
    bf.open()
    java_reader = bf._java_reader
    ref = weakref.ref(java_reader)

    del bf
    gc.collect()

    # Finalizer called reader.close() (full) which nulls currentId
    assert java_reader
    assert java_reader.getCurrentFile() is None
    del java_reader
    gc.collect()
    assert ref() is None


def test_gc_finalizer_from_suspended(simple_file: Path) -> None:
    """del bf from SUSPENDED triggers full cleanup (currentId nulled)."""
    bf = BioFile(simple_file)
    bf.open()
    java_reader = bf._java_reader

    bf.close()
    # close(true) preserves currentId
    assert java_reader
    assert java_reader.getCurrentFile() is not None

    del bf
    gc.collect()

    # Finalizer did full close -> currentId nulled
    assert java_reader.getCurrentFile() is None


# ---------------------------------------------------------------------------
# Memoization interaction
# ---------------------------------------------------------------------------


def test_memoize_suspend_resume_and_destroy(simple_file: Path, memo_dir: Path) -> None:
    """Memoizer is bypassed on suspend/resume; only matters for full re-init."""
    bf = BioFile(simple_file, memoize=1)
    bf.open()
    reader_first = bf._java_reader

    # Suspend/resume: same reader, Memoizer not involved
    bf.close()
    bf.open()
    assert bf._java_reader is reader_first
    _assert_open(bf)

    # Destroy + reopen: new reader (may load from memo file)
    bf.destroy()
    bf.open()
    assert bf._java_reader is not reader_first
    _assert_open(bf)

    bf.destroy()
