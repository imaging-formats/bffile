import typing
from typing import Protocol

import java.lang
import java.util.concurrent

class Cleaner:
    @typing.overload
    @staticmethod
    def create() -> Cleaner: ...
    @typing.overload
    @staticmethod
    def create(
        threadFactory: java.util.concurrent.ThreadFactory | typing.Callable,
    ) -> Cleaner: ...
    def register(
        self, object: typing.Any, runnable: java.lang.Runnable | typing.Callable
    ) -> Cleaner.Cleanable: ...
    class Cleanable:
        def clean(self) -> None: ...

_Reference__T = typing.TypeVar("_Reference__T")  # <T>

class Reference(typing.Generic[_Reference__T]):
    def clear(self) -> None: ...
    def enqueue(self) -> bool: ...
    def get(self) -> _Reference__T: ...
    def isEnqueued(self) -> bool: ...
    @staticmethod
    def reachabilityFence(object: typing.Any) -> None: ...
    def refersTo(self, t: _Reference__T) -> bool: ...

_ReferenceQueue__T = typing.TypeVar("_ReferenceQueue__T")  # <T>

class ReferenceQueue(typing.Generic[_ReferenceQueue__T]):
    def __init__(self): ...
    def poll(self) -> Reference[_ReferenceQueue__T]: ...
    @typing.overload
    def remove(self) -> Reference[_ReferenceQueue__T]: ...
    @typing.overload
    def remove(self, long: int) -> Reference[_ReferenceQueue__T]: ...

_PhantomReference__T = typing.TypeVar("_PhantomReference__T")  # <T>

class PhantomReference(
    Reference[_PhantomReference__T], typing.Generic[_PhantomReference__T]
):
    def __init__(
        self,
        t: _PhantomReference__T,
        referenceQueue: ReferenceQueue[_PhantomReference__T],
    ): ...
    def get(self) -> _PhantomReference__T: ...

_SoftReference__T = typing.TypeVar("_SoftReference__T")  # <T>

class SoftReference(Reference[_SoftReference__T], typing.Generic[_SoftReference__T]):
    @typing.overload
    def __init__(self, t: _SoftReference__T): ...
    @typing.overload
    def __init__(
        self, t: _SoftReference__T, referenceQueue: ReferenceQueue[_SoftReference__T]
    ): ...
    def get(self) -> _SoftReference__T: ...

_WeakReference__T = typing.TypeVar("_WeakReference__T")  # <T>

class WeakReference(Reference[_WeakReference__T], typing.Generic[_WeakReference__T]):
    @typing.overload
    def __init__(self, t: _WeakReference__T): ...
    @typing.overload
    def __init__(
        self, t: _WeakReference__T, referenceQueue: ReferenceQueue[_WeakReference__T]
    ): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.lang.ref")``.

    Cleaner: type[Cleaner]
    PhantomReference: type[PhantomReference]
    Reference: type[Reference]
    ReferenceQueue: type[ReferenceQueue]
    SoftReference: type[SoftReference]
    WeakReference: type[WeakReference]
