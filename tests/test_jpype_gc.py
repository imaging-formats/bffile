"""bffile turns off JPype's coupling of the Java and Python garbage collectors."""

from __future__ import annotations

import gc
import threading
import time
from typing import TYPE_CHECKING

import _jpype

from bffile._java_stuff import disable_jpype_gc_bridge, start_jvm
from bffile._jimports import jimport

if TYPE_CHECKING:
    import pytest


def test_jpype_gc_hook_removed() -> None:
    start_jvm()
    assert _jpype._collect not in gc.callbacks


def test_java_gc_does_not_force_python_gc() -> None:
    """With the hook in place, every Java GC runs a full Python GC on JPype's
    reference-queue thread. Fails if a JPype release changes how the trigger
    is re-armed."""
    start_jvm()
    System = jimport("java.lang.System")
    main_thread = threading.get_ident()
    forced: list[int] = []

    def on_gc(phase: str, info: dict) -> None:
        if phase == "start" and info["generation"] == 2:
            if threading.get_ident() != main_thread:
                forced.append(1)

    # the first Java GC after the hook is removed may still force one
    System.gc()
    time.sleep(0.3)
    gc.callbacks.append(on_gc)
    try:
        for _ in range(3):
            System.gc()
            time.sleep(0.3)  # the reference-queue thread polls every 250 ms
    finally:
        gc.callbacks.remove(on_gc)
    assert not forced


def test_bff_jpype_gc_keeps_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    start_jvm()
    # JPype installs the hook twice when it starts
    gc.callbacks.extend([_jpype._collect, _jpype._collect])
    try:
        monkeypatch.setenv("BFF_JPYPE_GC", "1")
        disable_jpype_gc_bridge()
        assert _jpype._collect in gc.callbacks
        monkeypatch.delenv("BFF_JPYPE_GC")
        disable_jpype_gc_bridge()
        assert _jpype._collect not in gc.callbacks
    finally:
        while _jpype._collect in gc.callbacks:
            gc.callbacks.remove(_jpype._collect)
