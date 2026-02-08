import builtins
import typing
from typing import Protocol

import java
import java.io
import java.lang
import java.security
import java.time
import java.time.temporal
import java.util
import java.util.concurrent.atomic
import java.util.concurrent.locks
import java.util.function
import java.util.stream
import jpype

_BlockingQueue__E = typing.TypeVar("_BlockingQueue__E")  # <E>

class BlockingQueue(
    java.util.Queue[_BlockingQueue__E], typing.Generic[_BlockingQueue__E]
):
    def add(self, e: _BlockingQueue__E) -> bool: ...
    def contains(self, object: typing.Any) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_BlockingQueue__E]
        | typing.Sequence[_BlockingQueue__E]
        | set[_BlockingQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_BlockingQueue__E]
        | typing.Sequence[_BlockingQueue__E]
        | set[_BlockingQueue__E],
        int: int,
    ) -> int: ...
    def equals(self, object: typing.Any) -> bool: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def offer(self, e: _BlockingQueue__E) -> bool: ...
    @typing.overload
    def offer(self, e: _BlockingQueue__E, long: int, timeUnit: TimeUnit) -> bool: ...
    @typing.overload
    def poll(self) -> _BlockingQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _BlockingQueue__E: ...
    def put(self, e: _BlockingQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _BlockingQueue__E: ...
    def take(self) -> _BlockingQueue__E: ...

class BrokenBarrierException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

_Callable__V = typing.TypeVar("_Callable__V")  # <V>

class Callable(typing.Generic[_Callable__V]):
    def call(self) -> _Callable__V: ...

class CancellationException(java.lang.IllegalStateException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class CompletionException(java.lang.RuntimeException):
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...

_CompletionService__V = typing.TypeVar("_CompletionService__V")  # <V>

class CompletionService(typing.Generic[_CompletionService__V]):
    @typing.overload
    def poll(self) -> Future[_CompletionService__V]: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> Future[_CompletionService__V]: ...
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable, v: _CompletionService__V
    ) -> Future[_CompletionService__V]: ...
    @typing.overload
    def submit(
        self,
        callable: Callable[_CompletionService__V]
        | typing.Callable[[], _CompletionService__V],
    ) -> Future[_CompletionService__V]: ...
    def take(self) -> Future[_CompletionService__V]: ...

_CompletionStage__T = typing.TypeVar("_CompletionStage__T")  # <T>

class CompletionStage(typing.Generic[_CompletionStage__T]):
    def acceptEither(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
    ) -> CompletionStage[None]: ...
    @typing.overload
    def acceptEitherAsync(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
    ) -> CompletionStage[None]: ...
    @typing.overload
    def acceptEitherAsync(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    _applyToEither__U = typing.TypeVar("_applyToEither__U")  # <U>
    def applyToEither(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        function: java.util.function.Function[_CompletionStage__T, _applyToEither__U]
        | typing.Callable[[_CompletionStage__T], _applyToEither__U],
    ) -> CompletionStage[_applyToEither__U]: ...
    _applyToEitherAsync_0__U = typing.TypeVar("_applyToEitherAsync_0__U")  # <U>
    _applyToEitherAsync_1__U = typing.TypeVar("_applyToEitherAsync_1__U")  # <U>
    @typing.overload
    def applyToEitherAsync(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        function: java.util.function.Function[
            _CompletionStage__T, _applyToEitherAsync_0__U
        ]
        | typing.Callable[[_CompletionStage__T], _applyToEitherAsync_0__U],
    ) -> CompletionStage[_applyToEitherAsync_0__U]: ...
    @typing.overload
    def applyToEitherAsync(
        self,
        completionStage: CompletionStage[_CompletionStage__T],
        function: java.util.function.Function[
            _CompletionStage__T, _applyToEitherAsync_1__U
        ]
        | typing.Callable[[_CompletionStage__T], _applyToEitherAsync_1__U],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_applyToEitherAsync_1__U]: ...
    def exceptionally(
        self,
        function: java.util.function.Function[java.lang.Throwable, _CompletionStage__T]
        | typing.Callable[[java.lang.Throwable], _CompletionStage__T],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def exceptionallyAsync(
        self,
        function: java.util.function.Function[java.lang.Throwable, _CompletionStage__T]
        | typing.Callable[[java.lang.Throwable], _CompletionStage__T],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def exceptionallyAsync(
        self,
        function: java.util.function.Function[java.lang.Throwable, _CompletionStage__T]
        | typing.Callable[[java.lang.Throwable], _CompletionStage__T],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_CompletionStage__T]: ...
    def exceptionallyCompose(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletionStage__T]
        ]
        | typing.Callable[[java.lang.Throwable], CompletionStage[_CompletionStage__T]],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def exceptionallyComposeAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletionStage__T]
        ]
        | typing.Callable[[java.lang.Throwable], CompletionStage[_CompletionStage__T]],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def exceptionallyComposeAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletionStage__T]
        ]
        | typing.Callable[[java.lang.Throwable], CompletionStage[_CompletionStage__T]],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_CompletionStage__T]: ...
    _handle__U = typing.TypeVar("_handle__U")  # <U>
    def handle(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, java.lang.Throwable, _handle__U
        ]
        | typing.Callable[[_CompletionStage__T, java.lang.Throwable], _handle__U],
    ) -> CompletionStage[_handle__U]: ...
    _handleAsync_0__U = typing.TypeVar("_handleAsync_0__U")  # <U>
    _handleAsync_1__U = typing.TypeVar("_handleAsync_1__U")  # <U>
    @typing.overload
    def handleAsync(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, java.lang.Throwable, _handleAsync_0__U
        ]
        | typing.Callable[
            [_CompletionStage__T, java.lang.Throwable], _handleAsync_0__U
        ],
    ) -> CompletionStage[_handleAsync_0__U]: ...
    @typing.overload
    def handleAsync(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, java.lang.Throwable, _handleAsync_1__U
        ]
        | typing.Callable[
            [_CompletionStage__T, java.lang.Throwable], _handleAsync_1__U
        ],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_handleAsync_1__U]: ...
    def runAfterBoth(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletionStage[None]: ...
    @typing.overload
    def runAfterBothAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletionStage[None]: ...
    @typing.overload
    def runAfterBothAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    def runAfterEither(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletionStage[None]: ...
    @typing.overload
    def runAfterEitherAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletionStage[None]: ...
    @typing.overload
    def runAfterEitherAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    def thenAccept(
        self,
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
    ) -> CompletionStage[None]: ...
    @typing.overload
    def thenAcceptAsync(
        self,
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
    ) -> CompletionStage[None]: ...
    @typing.overload
    def thenAcceptAsync(
        self,
        consumer: java.util.function.Consumer[_CompletionStage__T]
        | typing.Callable[[_CompletionStage__T], None],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    _thenAcceptBoth__U = typing.TypeVar("_thenAcceptBoth__U")  # <U>
    def thenAcceptBoth(
        self,
        completionStage: CompletionStage[_thenAcceptBoth__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, _thenAcceptBoth__U
        ]
        | typing.Callable[[_CompletionStage__T, _thenAcceptBoth__U], None],
    ) -> CompletionStage[None]: ...
    _thenAcceptBothAsync_0__U = typing.TypeVar("_thenAcceptBothAsync_0__U")  # <U>
    _thenAcceptBothAsync_1__U = typing.TypeVar("_thenAcceptBothAsync_1__U")  # <U>
    @typing.overload
    def thenAcceptBothAsync(
        self,
        completionStage: CompletionStage[_thenAcceptBothAsync_0__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, _thenAcceptBothAsync_0__U
        ]
        | typing.Callable[[_CompletionStage__T, _thenAcceptBothAsync_0__U], None],
    ) -> CompletionStage[None]: ...
    @typing.overload
    def thenAcceptBothAsync(
        self,
        completionStage: CompletionStage[_thenAcceptBothAsync_1__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, _thenAcceptBothAsync_1__U
        ]
        | typing.Callable[[_CompletionStage__T, _thenAcceptBothAsync_1__U], None],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    _thenApply__U = typing.TypeVar("_thenApply__U")  # <U>
    def thenApply(
        self,
        function: java.util.function.Function[_CompletionStage__T, _thenApply__U]
        | typing.Callable[[_CompletionStage__T], _thenApply__U],
    ) -> CompletionStage[_thenApply__U]: ...
    _thenApplyAsync_0__U = typing.TypeVar("_thenApplyAsync_0__U")  # <U>
    _thenApplyAsync_1__U = typing.TypeVar("_thenApplyAsync_1__U")  # <U>
    @typing.overload
    def thenApplyAsync(
        self,
        function: java.util.function.Function[_CompletionStage__T, _thenApplyAsync_0__U]
        | typing.Callable[[_CompletionStage__T], _thenApplyAsync_0__U],
    ) -> CompletionStage[_thenApplyAsync_0__U]: ...
    @typing.overload
    def thenApplyAsync(
        self,
        function: java.util.function.Function[_CompletionStage__T, _thenApplyAsync_1__U]
        | typing.Callable[[_CompletionStage__T], _thenApplyAsync_1__U],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_thenApplyAsync_1__U]: ...
    _thenCombine__U = typing.TypeVar("_thenCombine__U")  # <U>
    _thenCombine__V = typing.TypeVar("_thenCombine__V")  # <V>
    def thenCombine(
        self,
        completionStage: CompletionStage[_thenCombine__U],
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, _thenCombine__U, _thenCombine__V
        ]
        | typing.Callable[[_CompletionStage__T, _thenCombine__U], _thenCombine__V],
    ) -> CompletionStage[_thenCombine__V]: ...
    _thenCombineAsync_0__U = typing.TypeVar("_thenCombineAsync_0__U")  # <U>
    _thenCombineAsync_0__V = typing.TypeVar("_thenCombineAsync_0__V")  # <V>
    _thenCombineAsync_1__U = typing.TypeVar("_thenCombineAsync_1__U")  # <U>
    _thenCombineAsync_1__V = typing.TypeVar("_thenCombineAsync_1__V")  # <V>
    @typing.overload
    def thenCombineAsync(
        self,
        completionStage: CompletionStage[_thenCombineAsync_0__U],
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, _thenCombineAsync_0__U, _thenCombineAsync_0__V
        ]
        | typing.Callable[
            [_CompletionStage__T, _thenCombineAsync_0__U], _thenCombineAsync_0__V
        ],
    ) -> CompletionStage[_thenCombineAsync_0__V]: ...
    @typing.overload
    def thenCombineAsync(
        self,
        completionStage: CompletionStage[_thenCombineAsync_1__U],
        biFunction: java.util.function.BiFunction[
            _CompletionStage__T, _thenCombineAsync_1__U, _thenCombineAsync_1__V
        ]
        | typing.Callable[
            [_CompletionStage__T, _thenCombineAsync_1__U], _thenCombineAsync_1__V
        ],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_thenCombineAsync_1__V]: ...
    _thenCompose__U = typing.TypeVar("_thenCompose__U")  # <U>
    def thenCompose(
        self,
        function: java.util.function.Function[
            _CompletionStage__T, CompletionStage[_thenCompose__U]
        ]
        | typing.Callable[[_CompletionStage__T], CompletionStage[_thenCompose__U]],
    ) -> CompletionStage[_thenCompose__U]: ...
    _thenComposeAsync_0__U = typing.TypeVar("_thenComposeAsync_0__U")  # <U>
    _thenComposeAsync_1__U = typing.TypeVar("_thenComposeAsync_1__U")  # <U>
    @typing.overload
    def thenComposeAsync(
        self,
        function: java.util.function.Function[
            _CompletionStage__T, CompletionStage[_thenComposeAsync_0__U]
        ]
        | typing.Callable[
            [_CompletionStage__T], CompletionStage[_thenComposeAsync_0__U]
        ],
    ) -> CompletionStage[_thenComposeAsync_0__U]: ...
    @typing.overload
    def thenComposeAsync(
        self,
        function: java.util.function.Function[
            _CompletionStage__T, CompletionStage[_thenComposeAsync_1__U]
        ]
        | typing.Callable[
            [_CompletionStage__T], CompletionStage[_thenComposeAsync_1__U]
        ],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_thenComposeAsync_1__U]: ...
    def thenRun(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> CompletionStage[None]: ...
    @typing.overload
    def thenRunAsync(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> CompletionStage[None]: ...
    @typing.overload
    def thenRunAsync(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletionStage[None]: ...
    def toCompletableFuture(self) -> CompletableFuture[_CompletionStage__T]: ...
    def whenComplete(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletionStage__T, java.lang.Throwable], None],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def whenCompleteAsync(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletionStage__T, java.lang.Throwable], None],
    ) -> CompletionStage[_CompletionStage__T]: ...
    @typing.overload
    def whenCompleteAsync(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletionStage__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletionStage__T, java.lang.Throwable], None],
        executor: Executor | typing.Callable,
    ) -> CompletionStage[_CompletionStage__T]: ...

_ConcurrentLinkedDeque__E = typing.TypeVar("_ConcurrentLinkedDeque__E")  # <E>

class ConcurrentLinkedDeque(
    java.util.AbstractCollection[_ConcurrentLinkedDeque__E],
    java.util.Deque[_ConcurrentLinkedDeque__E],
    java.io.Serializable,
    typing.Generic[_ConcurrentLinkedDeque__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_ConcurrentLinkedDeque__E]
        | typing.Sequence[_ConcurrentLinkedDeque__E]
        | set[_ConcurrentLinkedDeque__E],
    ): ...
    def add(self, e: _ConcurrentLinkedDeque__E) -> bool: ...
    def addAll(
        self,
        collection: java.util.Collection[_ConcurrentLinkedDeque__E]
        | typing.Sequence[_ConcurrentLinkedDeque__E]
        | set[_ConcurrentLinkedDeque__E],
    ) -> bool: ...
    def addFirst(self, e: _ConcurrentLinkedDeque__E) -> None: ...
    def addLast(self, e: _ConcurrentLinkedDeque__E) -> None: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def descendingIterator(self) -> java.util.Iterator[_ConcurrentLinkedDeque__E]: ...
    def element(self) -> _ConcurrentLinkedDeque__E: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_ConcurrentLinkedDeque__E]
        | typing.Callable[[_ConcurrentLinkedDeque__E], None],
    ) -> None: ...
    def getFirst(self) -> _ConcurrentLinkedDeque__E: ...
    def getLast(self) -> _ConcurrentLinkedDeque__E: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_ConcurrentLinkedDeque__E]: ...
    def offer(self, e: _ConcurrentLinkedDeque__E) -> bool: ...
    def offerFirst(self, e: _ConcurrentLinkedDeque__E) -> bool: ...
    def offerLast(self, e: _ConcurrentLinkedDeque__E) -> bool: ...
    def peek(self) -> _ConcurrentLinkedDeque__E: ...
    def peekFirst(self) -> _ConcurrentLinkedDeque__E: ...
    def peekLast(self) -> _ConcurrentLinkedDeque__E: ...
    def poll(self) -> _ConcurrentLinkedDeque__E: ...
    def pollFirst(self) -> _ConcurrentLinkedDeque__E: ...
    def pollLast(self) -> _ConcurrentLinkedDeque__E: ...
    def pop(self) -> _ConcurrentLinkedDeque__E: ...
    def push(self, e: _ConcurrentLinkedDeque__E) -> None: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _ConcurrentLinkedDeque__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeFirst(self) -> _ConcurrentLinkedDeque__E: ...
    def removeFirstOccurrence(self, object: typing.Any) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_ConcurrentLinkedDeque__E]
        | typing.Callable[[_ConcurrentLinkedDeque__E], bool],
    ) -> bool: ...
    def removeLast(self) -> _ConcurrentLinkedDeque__E: ...
    def removeLastOccurrence(self, object: typing.Any) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_ConcurrentLinkedDeque__E]: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_ConcurrentLinkedQueue__E = typing.TypeVar("_ConcurrentLinkedQueue__E")  # <E>

class ConcurrentLinkedQueue(
    java.util.AbstractQueue[_ConcurrentLinkedQueue__E],
    java.util.Queue[_ConcurrentLinkedQueue__E],
    java.io.Serializable,
    typing.Generic[_ConcurrentLinkedQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_ConcurrentLinkedQueue__E]
        | typing.Sequence[_ConcurrentLinkedQueue__E]
        | set[_ConcurrentLinkedQueue__E],
    ): ...
    def add(self, e: _ConcurrentLinkedQueue__E) -> bool: ...
    def addAll(
        self,
        collection: java.util.Collection[_ConcurrentLinkedQueue__E]
        | typing.Sequence[_ConcurrentLinkedQueue__E]
        | set[_ConcurrentLinkedQueue__E],
    ) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_ConcurrentLinkedQueue__E]
        | typing.Callable[[_ConcurrentLinkedQueue__E], None],
    ) -> None: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_ConcurrentLinkedQueue__E]: ...
    def offer(self, e: _ConcurrentLinkedQueue__E) -> bool: ...
    def peek(self) -> _ConcurrentLinkedQueue__E: ...
    def poll(self) -> _ConcurrentLinkedQueue__E: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _ConcurrentLinkedQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_ConcurrentLinkedQueue__E]
        | typing.Callable[[_ConcurrentLinkedQueue__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_ConcurrentLinkedQueue__E]: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_ConcurrentMap__K = typing.TypeVar("_ConcurrentMap__K")  # <K>
_ConcurrentMap__V = typing.TypeVar("_ConcurrentMap__V")  # <V>

class ConcurrentMap(
    java.util.Map[_ConcurrentMap__K, _ConcurrentMap__V],
    typing.Generic[_ConcurrentMap__K, _ConcurrentMap__V],
):
    def compute(
        self,
        k: _ConcurrentMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentMap__K, _ConcurrentMap__V, _ConcurrentMap__V
        ]
        | typing.Callable[[_ConcurrentMap__K, _ConcurrentMap__V], _ConcurrentMap__V],
    ) -> _ConcurrentMap__V: ...
    def computeIfAbsent(
        self,
        k: _ConcurrentMap__K,
        function: java.util.function.Function[_ConcurrentMap__K, _ConcurrentMap__V]
        | typing.Callable[[_ConcurrentMap__K], _ConcurrentMap__V],
    ) -> _ConcurrentMap__V: ...
    def computeIfPresent(
        self,
        k: _ConcurrentMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentMap__K, _ConcurrentMap__V, _ConcurrentMap__V
        ]
        | typing.Callable[[_ConcurrentMap__K, _ConcurrentMap__V], _ConcurrentMap__V],
    ) -> _ConcurrentMap__V: ...
    def equals(self, object: typing.Any) -> bool: ...
    def forEach(
        self,
        biConsumer: java.util.function.BiConsumer[_ConcurrentMap__K, _ConcurrentMap__V]
        | typing.Callable[[_ConcurrentMap__K, _ConcurrentMap__V], None],
    ) -> None: ...
    def getOrDefault(
        self, object: typing.Any, v: _ConcurrentMap__V
    ) -> _ConcurrentMap__V: ...
    def hashCode(self) -> int: ...
    def merge(
        self,
        k: _ConcurrentMap__K,
        v: _ConcurrentMap__V,
        biFunction: java.util.function.BiFunction[
            _ConcurrentMap__V, _ConcurrentMap__V, _ConcurrentMap__V
        ]
        | typing.Callable[[_ConcurrentMap__V, _ConcurrentMap__V], _ConcurrentMap__V],
    ) -> _ConcurrentMap__V: ...
    def putIfAbsent(
        self, k: _ConcurrentMap__K, v: _ConcurrentMap__V
    ) -> _ConcurrentMap__V: ...
    @typing.overload
    def remove(self, object: typing.Any, object2: typing.Any) -> bool: ...
    @typing.overload
    def remove(self, object: typing.Any) -> _ConcurrentMap__V: ...
    @typing.overload
    def replace(
        self, k: _ConcurrentMap__K, v: _ConcurrentMap__V, v2: _ConcurrentMap__V
    ) -> bool: ...
    @typing.overload
    def replace(
        self, k: _ConcurrentMap__K, v: _ConcurrentMap__V
    ) -> _ConcurrentMap__V: ...
    def replaceAll(
        self,
        biFunction: java.util.function.BiFunction[
            _ConcurrentMap__K, _ConcurrentMap__V, _ConcurrentMap__V
        ]
        | typing.Callable[[_ConcurrentMap__K, _ConcurrentMap__V], _ConcurrentMap__V],
    ) -> None: ...

_ConcurrentSkipListSet__E = typing.TypeVar("_ConcurrentSkipListSet__E")  # <E>

class ConcurrentSkipListSet(
    java.util.AbstractSet[_ConcurrentSkipListSet__E],
    java.util.NavigableSet[_ConcurrentSkipListSet__E],
    java.lang.Cloneable,
    java.io.Serializable,
    typing.Generic[_ConcurrentSkipListSet__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_ConcurrentSkipListSet__E]
        | typing.Sequence[_ConcurrentSkipListSet__E]
        | set[_ConcurrentSkipListSet__E],
    ): ...
    @typing.overload
    def __init__(
        self,
        comparator: java.util.Comparator[_ConcurrentSkipListSet__E]
        | typing.Callable[[_ConcurrentSkipListSet__E, _ConcurrentSkipListSet__E], int],
    ): ...
    @typing.overload
    def __init__(self, sortedSet: java.util.SortedSet[_ConcurrentSkipListSet__E]): ...
    def add(self, e: _ConcurrentSkipListSet__E) -> bool: ...
    def addFirst(self, e: _ConcurrentSkipListSet__E) -> None: ...
    def addLast(self, e: _ConcurrentSkipListSet__E) -> None: ...
    def ceiling(self, e: _ConcurrentSkipListSet__E) -> _ConcurrentSkipListSet__E: ...
    def clear(self) -> None: ...
    def clone(self) -> ConcurrentSkipListSet[_ConcurrentSkipListSet__E]: ...
    def comparator(self) -> java.util.Comparator[_ConcurrentSkipListSet__E]: ...
    def contains(self, object: typing.Any) -> bool: ...
    def descendingIterator(self) -> java.util.Iterator[_ConcurrentSkipListSet__E]: ...
    def descendingSet(self) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    def equals(self, object: typing.Any) -> bool: ...
    def first(self) -> _ConcurrentSkipListSet__E: ...
    def floor(self, e: _ConcurrentSkipListSet__E) -> _ConcurrentSkipListSet__E: ...
    @typing.overload
    def headSet(
        self, e: _ConcurrentSkipListSet__E
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    @typing.overload
    def headSet(
        self, e: _ConcurrentSkipListSet__E, boolean: bool
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    def higher(self, e: _ConcurrentSkipListSet__E) -> _ConcurrentSkipListSet__E: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_ConcurrentSkipListSet__E]: ...
    def last(self) -> _ConcurrentSkipListSet__E: ...
    def lower(self, e: _ConcurrentSkipListSet__E) -> _ConcurrentSkipListSet__E: ...
    def pollFirst(self) -> _ConcurrentSkipListSet__E: ...
    def pollLast(self) -> _ConcurrentSkipListSet__E: ...
    def remove(self, object: typing.Any) -> bool: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_ConcurrentSkipListSet__E]: ...
    @typing.overload
    def subSet(
        self,
        e: _ConcurrentSkipListSet__E,
        boolean: bool,
        e2: _ConcurrentSkipListSet__E,
        boolean2: bool,
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    @typing.overload
    def subSet(
        self, e: _ConcurrentSkipListSet__E, e2: _ConcurrentSkipListSet__E
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    @typing.overload
    def tailSet(
        self, e: _ConcurrentSkipListSet__E
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...
    @typing.overload
    def tailSet(
        self, e: _ConcurrentSkipListSet__E, boolean: bool
    ) -> java.util.NavigableSet[_ConcurrentSkipListSet__E]: ...

_CopyOnWriteArrayList__E = typing.TypeVar("_CopyOnWriteArrayList__E")  # <E>

class CopyOnWriteArrayList(
    java.util.List[_CopyOnWriteArrayList__E],
    java.util.RandomAccess,
    java.lang.Cloneable,
    java.io.Serializable,
    typing.Generic[_CopyOnWriteArrayList__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, eArray: list[_CopyOnWriteArrayList__E] | jpype.JArray): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_CopyOnWriteArrayList__E]
        | typing.Sequence[_CopyOnWriteArrayList__E]
        | builtins.set[_CopyOnWriteArrayList__E],
    ): ...
    @typing.overload
    def add(self, e: _CopyOnWriteArrayList__E) -> bool: ...
    @typing.overload
    def add(self, int: int, e: _CopyOnWriteArrayList__E) -> None: ...
    @typing.overload
    def addAll(
        self,
        int: int,
        collection: java.util.Collection[_CopyOnWriteArrayList__E]
        | typing.Sequence[_CopyOnWriteArrayList__E]
        | builtins.set[_CopyOnWriteArrayList__E],
    ) -> bool: ...
    @typing.overload
    def addAll(
        self,
        collection: java.util.Collection[_CopyOnWriteArrayList__E]
        | typing.Sequence[_CopyOnWriteArrayList__E]
        | builtins.set[_CopyOnWriteArrayList__E],
    ) -> bool: ...
    def addAllAbsent(
        self,
        collection: java.util.Collection[_CopyOnWriteArrayList__E]
        | typing.Sequence[_CopyOnWriteArrayList__E]
        | builtins.set[_CopyOnWriteArrayList__E],
    ) -> int: ...
    def addFirst(self, e: _CopyOnWriteArrayList__E) -> None: ...
    def addIfAbsent(self, e: _CopyOnWriteArrayList__E) -> bool: ...
    def addLast(self, e: _CopyOnWriteArrayList__E) -> None: ...
    def clear(self) -> None: ...
    def clone(self) -> typing.Any: ...
    def contains(self, object: typing.Any) -> bool: ...
    def containsAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | builtins.set[typing.Any],
    ) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_CopyOnWriteArrayList__E]
        | typing.Callable[[_CopyOnWriteArrayList__E], None],
    ) -> None: ...
    def get(self, int: int) -> _CopyOnWriteArrayList__E: ...
    def getFirst(self) -> _CopyOnWriteArrayList__E: ...
    def getLast(self) -> _CopyOnWriteArrayList__E: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def indexOf(self, object: typing.Any) -> int: ...
    @typing.overload
    def indexOf(self, e: _CopyOnWriteArrayList__E, int: int) -> int: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_CopyOnWriteArrayList__E]: ...
    @typing.overload
    def lastIndexOf(self, object: typing.Any) -> int: ...
    @typing.overload
    def lastIndexOf(self, e: _CopyOnWriteArrayList__E, int: int) -> int: ...
    @typing.overload
    def listIterator(self) -> java.util.ListIterator[_CopyOnWriteArrayList__E]: ...
    @typing.overload
    def listIterator(
        self, int: int
    ) -> java.util.ListIterator[_CopyOnWriteArrayList__E]: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | builtins.set[typing.Any],
    ) -> bool: ...
    def removeFirst(self) -> _CopyOnWriteArrayList__E: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_CopyOnWriteArrayList__E]
        | typing.Callable[[_CopyOnWriteArrayList__E], bool],
    ) -> bool: ...
    def removeLast(self) -> _CopyOnWriteArrayList__E: ...
    def replaceAll(
        self,
        unaryOperator: java.util.function.UnaryOperator[_CopyOnWriteArrayList__E]
        | typing.Callable,
    ) -> None: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | builtins.set[typing.Any],
    ) -> bool: ...
    def reversed(self) -> java.util.List[_CopyOnWriteArrayList__E]: ...
    def set(
        self, int: int, e: _CopyOnWriteArrayList__E
    ) -> _CopyOnWriteArrayList__E: ...
    def size(self) -> int: ...
    def sort(
        self,
        comparator: java.util.Comparator[_CopyOnWriteArrayList__E]
        | typing.Callable[[_CopyOnWriteArrayList__E, _CopyOnWriteArrayList__E], int],
    ) -> None: ...
    def spliterator(self) -> java.util.Spliterator[_CopyOnWriteArrayList__E]: ...
    def subList(
        self, int: int, int2: int
    ) -> java.util.List[_CopyOnWriteArrayList__E]: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_CopyOnWriteArraySet__E = typing.TypeVar("_CopyOnWriteArraySet__E")  # <E>

class CopyOnWriteArraySet(
    java.util.AbstractSet[_CopyOnWriteArraySet__E],
    java.io.Serializable,
    typing.Generic[_CopyOnWriteArraySet__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_CopyOnWriteArraySet__E]
        | typing.Sequence[_CopyOnWriteArraySet__E]
        | set[_CopyOnWriteArraySet__E],
    ): ...
    def add(self, e: _CopyOnWriteArraySet__E) -> bool: ...
    def addAll(
        self,
        collection: java.util.Collection[_CopyOnWriteArraySet__E]
        | typing.Sequence[_CopyOnWriteArraySet__E]
        | set[_CopyOnWriteArraySet__E],
    ) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def containsAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_CopyOnWriteArraySet__E]
        | typing.Callable[[_CopyOnWriteArraySet__E], None],
    ) -> None: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_CopyOnWriteArraySet__E]: ...
    def remove(self, object: typing.Any) -> bool: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_CopyOnWriteArraySet__E]
        | typing.Callable[[_CopyOnWriteArraySet__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_CopyOnWriteArraySet__E]: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...

class CountDownLatch:
    def __init__(self, int: int): ...
    @typing.overload
    def await_(self, long: int, timeUnit: TimeUnit) -> bool: ...
    @typing.overload
    def await_(self) -> None: ...
    def countDown(self) -> None: ...
    def getCount(self) -> int: ...
    def toString(self) -> java.lang.String: ...

class CyclicBarrier:
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, runnable: java.lang.Runnable | typing.Callable): ...
    @typing.overload
    def await_(self) -> int: ...
    @typing.overload
    def await_(self, long: int, timeUnit: TimeUnit) -> int: ...
    def getNumberWaiting(self) -> int: ...
    def getParties(self) -> int: ...
    def isBroken(self) -> bool: ...
    def reset(self) -> None: ...

class Delayed(java.lang.Comparable["Delayed"]):
    def getDelay(self, timeUnit: TimeUnit) -> int: ...

_Exchanger__V = typing.TypeVar("_Exchanger__V")  # <V>

class Exchanger(typing.Generic[_Exchanger__V]):
    def __init__(self): ...
    @typing.overload
    def exchange(self, v: _Exchanger__V) -> _Exchanger__V: ...
    @typing.overload
    def exchange(
        self, v: _Exchanger__V, long: int, timeUnit: TimeUnit
    ) -> _Exchanger__V: ...

class ExecutionException(java.lang.Exception):
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...

class Executor:
    def execute(self, runnable: java.lang.Runnable | typing.Callable) -> None: ...

class Executors:
    _callable_1__T = typing.TypeVar("_callable_1__T")  # <T>
    @typing.overload
    @staticmethod
    def callable(
        runnable: java.lang.Runnable | typing.Callable,
    ) -> Callable[typing.Any]: ...
    @typing.overload
    @staticmethod
    def callable(
        runnable: java.lang.Runnable | typing.Callable, t: _callable_1__T
    ) -> Callable[_callable_1__T]: ...
    @typing.overload
    @staticmethod
    def callable(
        privilegedAction: java.security.PrivilegedAction[typing.Any]
        | typing.Callable[[], typing.Any],
    ) -> Callable[typing.Any]: ...
    @typing.overload
    @staticmethod
    def callable(
        privilegedExceptionAction: java.security.PrivilegedExceptionAction[typing.Any]
        | typing.Callable[[], typing.Any],
    ) -> Callable[typing.Any]: ...
    @staticmethod
    def defaultThreadFactory() -> ThreadFactory: ...
    @typing.overload
    @staticmethod
    def newCachedThreadPool() -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newCachedThreadPool(
        threadFactory: ThreadFactory | typing.Callable,
    ) -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newFixedThreadPool(int: int) -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newFixedThreadPool(
        int: int, threadFactory: ThreadFactory | typing.Callable
    ) -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newScheduledThreadPool(int: int) -> ScheduledExecutorService: ...
    @typing.overload
    @staticmethod
    def newScheduledThreadPool(
        int: int, threadFactory: ThreadFactory | typing.Callable
    ) -> ScheduledExecutorService: ...
    @typing.overload
    @staticmethod
    def newSingleThreadExecutor() -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newSingleThreadExecutor(
        threadFactory: ThreadFactory | typing.Callable,
    ) -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newSingleThreadScheduledExecutor() -> ScheduledExecutorService: ...
    @typing.overload
    @staticmethod
    def newSingleThreadScheduledExecutor(
        threadFactory: ThreadFactory | typing.Callable,
    ) -> ScheduledExecutorService: ...
    @staticmethod
    def newThreadPerTaskExecutor(
        threadFactory: ThreadFactory | typing.Callable,
    ) -> ExecutorService: ...
    @staticmethod
    def newVirtualThreadPerTaskExecutor() -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newWorkStealingPool() -> ExecutorService: ...
    @typing.overload
    @staticmethod
    def newWorkStealingPool(int: int) -> ExecutorService: ...
    _privilegedCallable__T = typing.TypeVar("_privilegedCallable__T")  # <T>
    @staticmethod
    def privilegedCallable(
        callable: Callable[_privilegedCallable__T]
        | typing.Callable[[], _privilegedCallable__T],
    ) -> Callable[_privilegedCallable__T]: ...
    _privilegedCallableUsingCurrentClassLoader__T = typing.TypeVar(
        "_privilegedCallableUsingCurrentClassLoader__T"
    )  # <T>
    @staticmethod
    def privilegedCallableUsingCurrentClassLoader(
        callable: Callable[_privilegedCallableUsingCurrentClassLoader__T]
        | typing.Callable[[], _privilegedCallableUsingCurrentClassLoader__T],
    ) -> Callable[_privilegedCallableUsingCurrentClassLoader__T]: ...
    @staticmethod
    def privilegedThreadFactory() -> ThreadFactory: ...
    @staticmethod
    def unconfigurableExecutorService(
        executorService: ExecutorService,
    ) -> ExecutorService: ...
    @staticmethod
    def unconfigurableScheduledExecutorService(
        scheduledExecutorService: ScheduledExecutorService,
    ) -> ScheduledExecutorService: ...

class ForkJoinWorkerThread(java.lang.Thread):
    def getPool(self) -> ForkJoinPool: ...
    def getPoolIndex(self) -> int: ...
    def getQueuedTaskCount(self) -> int: ...
    def run(self) -> None: ...

_Future__V = typing.TypeVar("_Future__V")  # <V>

class Future(typing.Generic[_Future__V]):
    def cancel(self, boolean: bool) -> bool: ...
    def exceptionNow(self) -> java.lang.Throwable: ...
    @typing.overload
    def get(self) -> _Future__V: ...
    @typing.overload
    def get(self, long: int, timeUnit: TimeUnit) -> _Future__V: ...
    def isCancelled(self) -> bool: ...
    def isDone(self) -> bool: ...
    def resultNow(self) -> _Future__V: ...
    def state(self) -> Future.State: ...
    class State(java.lang.Enum["Future.State"]):
        RUNNING: typing.ClassVar[Future.State] = ...
        SUCCESS: typing.ClassVar[Future.State] = ...
        FAILED: typing.ClassVar[Future.State] = ...
        CANCELLED: typing.ClassVar[Future.State] = ...
        _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_0__T], string: java.lang.String | str
        ) -> _valueOf_0__T: ...
        @typing.overload
        @staticmethod
        def valueOf(string: java.lang.String | str) -> Future.State: ...
        @staticmethod
        def values() -> typing.MutableSequence[Future.State]: ...

class Phaser:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, phaser: Phaser): ...
    @typing.overload
    def __init__(self, phaser: Phaser, int: int): ...
    def arrive(self) -> int: ...
    def arriveAndAwaitAdvance(self) -> int: ...
    def arriveAndDeregister(self) -> int: ...
    def awaitAdvance(self, int: int) -> int: ...
    @typing.overload
    def awaitAdvanceInterruptibly(self, int: int) -> int: ...
    @typing.overload
    def awaitAdvanceInterruptibly(
        self, int: int, long: int, timeUnit: TimeUnit
    ) -> int: ...
    def bulkRegister(self, int: int) -> int: ...
    def forceTermination(self) -> None: ...
    def getArrivedParties(self) -> int: ...
    def getParent(self) -> Phaser: ...
    def getPhase(self) -> int: ...
    def getRegisteredParties(self) -> int: ...
    def getRoot(self) -> Phaser: ...
    def getUnarrivedParties(self) -> int: ...
    def isTerminated(self) -> bool: ...
    def register(self) -> int: ...
    def toString(self) -> java.lang.String: ...

class RejectedExecutionException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...

class RejectedExecutionHandler:
    def rejectedExecution(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        threadPoolExecutor: ThreadPoolExecutor,
    ) -> None: ...

class Semaphore(java.io.Serializable):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, boolean: bool): ...
    @typing.overload
    def acquire(self) -> None: ...
    @typing.overload
    def acquire(self, int: int) -> None: ...
    @typing.overload
    def acquireUninterruptibly(self) -> None: ...
    @typing.overload
    def acquireUninterruptibly(self, int: int) -> None: ...
    def availablePermits(self) -> int: ...
    def drainPermits(self) -> int: ...
    def getQueueLength(self) -> int: ...
    def hasQueuedThreads(self) -> bool: ...
    def isFair(self) -> bool: ...
    @typing.overload
    def release(self) -> None: ...
    @typing.overload
    def release(self, int: int) -> None: ...
    def toString(self) -> java.lang.String: ...
    @typing.overload
    def tryAcquire(self) -> bool: ...
    @typing.overload
    def tryAcquire(self, int: int) -> bool: ...
    @typing.overload
    def tryAcquire(self, int: int, long: int, timeUnit: TimeUnit) -> bool: ...
    @typing.overload
    def tryAcquire(self, long: int, timeUnit: TimeUnit) -> bool: ...

class StructureViolationException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

_StructuredTaskScope__Joiner__T = typing.TypeVar(
    "_StructuredTaskScope__Joiner__T"
)  # <T>
_StructuredTaskScope__Joiner__R = typing.TypeVar(
    "_StructuredTaskScope__Joiner__R"
)  # <R>
_StructuredTaskScope__Subtask__T = typing.TypeVar(
    "_StructuredTaskScope__Subtask__T"
)  # <T>
_StructuredTaskScope__T = typing.TypeVar("_StructuredTaskScope__T")  # <T>
_StructuredTaskScope__R = typing.TypeVar("_StructuredTaskScope__R")  # <R>

class StructuredTaskScope(
    java.lang.AutoCloseable,
    typing.Generic[_StructuredTaskScope__T, _StructuredTaskScope__R],
):
    def close(self) -> None: ...
    _fork_0__U = typing.TypeVar("_fork_0__U")  # <U>
    _fork_1__U = typing.TypeVar("_fork_1__U")  # <U>
    @typing.overload
    def fork(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> StructuredTaskScope.Subtask[_fork_0__U]: ...
    @typing.overload
    def fork(
        self, callable: Callable[_fork_1__U] | typing.Callable[[], _fork_1__U]
    ) -> StructuredTaskScope.Subtask[_fork_1__U]: ...
    def isCancelled(self) -> bool: ...
    def join(self) -> _StructuredTaskScope__R: ...
    _open_0__T = typing.TypeVar("_open_0__T")  # <T>
    _open_1__T = typing.TypeVar("_open_1__T")  # <T>
    _open_1__R = typing.TypeVar("_open_1__R")  # <R>
    _open_2__T = typing.TypeVar("_open_2__T")  # <T>
    _open_2__R = typing.TypeVar("_open_2__R")  # <R>
    @typing.overload
    @staticmethod
    def open() -> StructuredTaskScope[_open_0__T, None]: ...
    @typing.overload
    @staticmethod
    def open(
        joiner: StructuredTaskScope.Joiner[_open_1__T, _open_1__R]
        | typing.Callable[[], _open_1__R],
    ) -> StructuredTaskScope[_open_1__T, _open_1__R]: ...
    @typing.overload
    @staticmethod
    def open(
        joiner: StructuredTaskScope.Joiner[_open_2__T, _open_2__R]
        | typing.Callable[[], _open_2__R],
        function: java.util.function.Function[
            StructuredTaskScope.Configuration, StructuredTaskScope.Configuration
        ]
        | typing.Callable[
            [StructuredTaskScope.Configuration], StructuredTaskScope.Configuration
        ],
    ) -> StructuredTaskScope[_open_2__T, _open_2__R]: ...
    class Configuration:
        def withName(
            self, string: java.lang.String | str
        ) -> StructuredTaskScope.Configuration: ...
        def withThreadFactory(
            self, threadFactory: ThreadFactory | typing.Callable
        ) -> StructuredTaskScope.Configuration: ...
        def withTimeout(
            self, duration: java.time.Duration
        ) -> StructuredTaskScope.Configuration: ...

    class FailedException(java.lang.RuntimeException): ...

    class Joiner(
        typing.Generic[_StructuredTaskScope__Joiner__T, _StructuredTaskScope__Joiner__R]
    ):
        _allSuccessfulOrThrow__T = typing.TypeVar("_allSuccessfulOrThrow__T")  # <T>
        @staticmethod
        def allSuccessfulOrThrow() -> StructuredTaskScope.Joiner[
            _allSuccessfulOrThrow__T,
            java.util.stream.Stream[
                StructuredTaskScope.Subtask[_allSuccessfulOrThrow__T]
            ],
        ]: ...
        _allUntil__T = typing.TypeVar("_allUntil__T")  # <T>
        @staticmethod
        def allUntil(
            predicate: java.util.function.Predicate[
                StructuredTaskScope.Subtask[_allUntil__T]
            ]
            | typing.Callable[[StructuredTaskScope.Subtask[_allUntil__T]], bool],
        ) -> StructuredTaskScope.Joiner[
            _allUntil__T,
            java.util.stream.Stream[StructuredTaskScope.Subtask[_allUntil__T]],
        ]: ...
        _anySuccessfulResultOrThrow__T = typing.TypeVar(
            "_anySuccessfulResultOrThrow__T"
        )  # <T>
        @staticmethod
        def anySuccessfulResultOrThrow() -> StructuredTaskScope.Joiner[
            _anySuccessfulResultOrThrow__T, _anySuccessfulResultOrThrow__T
        ]: ...
        _awaitAll__T = typing.TypeVar("_awaitAll__T")  # <T>
        @staticmethod
        def awaitAll() -> StructuredTaskScope.Joiner[_awaitAll__T, None]: ...
        _awaitAllSuccessfulOrThrow__T = typing.TypeVar(
            "_awaitAllSuccessfulOrThrow__T"
        )  # <T>
        @staticmethod
        def awaitAllSuccessfulOrThrow() -> StructuredTaskScope.Joiner[
            _awaitAllSuccessfulOrThrow__T, None
        ]: ...
        def onComplete(
            self, subtask: StructuredTaskScope.Subtask[_StructuredTaskScope__Joiner__T]
        ) -> bool: ...
        def onFork(
            self, subtask: StructuredTaskScope.Subtask[_StructuredTaskScope__Joiner__T]
        ) -> bool: ...
        def result(self) -> _StructuredTaskScope__Joiner__R: ...

    class Subtask(
        java.util.function.Supplier[_StructuredTaskScope__Subtask__T],
        typing.Generic[_StructuredTaskScope__Subtask__T],
    ):
        def exception(self) -> java.lang.Throwable: ...
        def get(self) -> _StructuredTaskScope__Subtask__T: ...
        def state(self) -> StructuredTaskScope.Subtask.State: ...
        class State(java.lang.Enum["StructuredTaskScope.Subtask.State"]):
            UNAVAILABLE: typing.ClassVar[StructuredTaskScope.Subtask.State] = ...
            SUCCESS: typing.ClassVar[StructuredTaskScope.Subtask.State] = ...
            FAILED: typing.ClassVar[StructuredTaskScope.Subtask.State] = ...
            _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
            @typing.overload
            @staticmethod
            def valueOf(
                class_: type[_valueOf_0__T], string: java.lang.String | str
            ) -> _valueOf_0__T: ...
            @typing.overload
            @staticmethod
            def valueOf(
                string: java.lang.String | str,
            ) -> StructuredTaskScope.Subtask.State: ...
            @staticmethod
            def values() -> typing.MutableSequence[
                StructuredTaskScope.Subtask.State
            ]: ...

    class TimeoutException(java.lang.RuntimeException): ...

class ThreadFactory:
    def newThread(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> java.lang.Thread: ...

class ThreadLocalRandom(java.util.Random):
    @staticmethod
    def current() -> ThreadLocalRandom: ...
    @typing.overload
    def doubles(self) -> java.util.stream.DoubleStream: ...
    @typing.overload
    def doubles(
        self, double: float, double2: float
    ) -> java.util.stream.DoubleStream: ...
    @typing.overload
    def doubles(self, long: int) -> java.util.stream.DoubleStream: ...
    @typing.overload
    def doubles(
        self, long: int, double: float, double2: float
    ) -> java.util.stream.DoubleStream: ...
    @typing.overload
    def ints(self) -> java.util.stream.IntStream: ...
    @typing.overload
    def ints(self, int: int, int2: int) -> java.util.stream.IntStream: ...
    @typing.overload
    def ints(self, long: int) -> java.util.stream.IntStream: ...
    @typing.overload
    def ints(self, long: int, int: int, int2: int) -> java.util.stream.IntStream: ...
    @typing.overload
    def longs(self) -> java.util.stream.LongStream: ...
    @typing.overload
    def longs(self, long: int) -> java.util.stream.LongStream: ...
    @typing.overload
    def longs(self, long: int, long2: int) -> java.util.stream.LongStream: ...
    @typing.overload
    def longs(
        self, long: int, long2: int, long3: int
    ) -> java.util.stream.LongStream: ...
    def nextBoolean(self) -> bool: ...
    @typing.overload
    def nextDouble(self) -> float: ...
    @typing.overload
    def nextDouble(self, double: float) -> float: ...
    @typing.overload
    def nextDouble(self, double: float, double2: float) -> float: ...
    @typing.overload
    def nextFloat(self) -> float: ...
    @typing.overload
    def nextFloat(self, float: float) -> float: ...
    @typing.overload
    def nextFloat(self, float: float, float2: float) -> float: ...
    @typing.overload
    def nextInt(self) -> int: ...
    @typing.overload
    def nextInt(self, int: int) -> int: ...
    @typing.overload
    def nextInt(self, int: int, int2: int) -> int: ...
    @typing.overload
    def nextLong(self) -> int: ...
    @typing.overload
    def nextLong(self, long: int) -> int: ...
    @typing.overload
    def nextLong(self, long: int, long2: int) -> int: ...
    def setSeed(self, long: int) -> None: ...

class TimeUnit(java.lang.Enum["TimeUnit"]):
    NANOSECONDS: typing.ClassVar[TimeUnit] = ...
    MICROSECONDS: typing.ClassVar[TimeUnit] = ...
    MILLISECONDS: typing.ClassVar[TimeUnit] = ...
    SECONDS: typing.ClassVar[TimeUnit] = ...
    MINUTES: typing.ClassVar[TimeUnit] = ...
    HOURS: typing.ClassVar[TimeUnit] = ...
    DAYS: typing.ClassVar[TimeUnit] = ...
    @typing.overload
    def convert(self, duration: java.time.Duration) -> int: ...
    @typing.overload
    def convert(self, long: int, timeUnit: TimeUnit) -> int: ...
    @staticmethod
    def of(chronoUnit: java.time.temporal.ChronoUnit) -> TimeUnit: ...
    def sleep(self, long: int) -> None: ...
    def timedJoin(self, thread: java.lang.Thread, long: int) -> None: ...
    def timedWait(self, object: typing.Any, long: int) -> None: ...
    def toChronoUnit(self) -> java.time.temporal.ChronoUnit: ...
    def toDays(self, long: int) -> int: ...
    def toHours(self, long: int) -> int: ...
    def toMicros(self, long: int) -> int: ...
    def toMillis(self, long: int) -> int: ...
    def toMinutes(self, long: int) -> int: ...
    def toNanos(self, long: int) -> int: ...
    def toSeconds(self, long: int) -> int: ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> TimeUnit: ...
    @staticmethod
    def values() -> typing.MutableSequence[TimeUnit]: ...

class TimeoutException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

_ArrayBlockingQueue__E = typing.TypeVar("_ArrayBlockingQueue__E")  # <E>

class ArrayBlockingQueue(
    java.util.AbstractQueue[_ArrayBlockingQueue__E],
    BlockingQueue[_ArrayBlockingQueue__E],
    java.io.Serializable,
    typing.Generic[_ArrayBlockingQueue__E],
):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, boolean: bool): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        boolean: bool,
        collection: java.util.Collection[_ArrayBlockingQueue__E]
        | typing.Sequence[_ArrayBlockingQueue__E]
        | set[_ArrayBlockingQueue__E],
    ): ...
    def add(self, e: _ArrayBlockingQueue__E) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_ArrayBlockingQueue__E]
        | typing.Sequence[_ArrayBlockingQueue__E]
        | set[_ArrayBlockingQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_ArrayBlockingQueue__E]
        | typing.Sequence[_ArrayBlockingQueue__E]
        | set[_ArrayBlockingQueue__E],
        int: int,
    ) -> int: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_ArrayBlockingQueue__E]
        | typing.Callable[[_ArrayBlockingQueue__E], None],
    ) -> None: ...
    def iterator(self) -> java.util.Iterator[_ArrayBlockingQueue__E]: ...
    @typing.overload
    def offer(self, e: _ArrayBlockingQueue__E) -> bool: ...
    @typing.overload
    def offer(
        self, e: _ArrayBlockingQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _ArrayBlockingQueue__E: ...
    @typing.overload
    def poll(self) -> _ArrayBlockingQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _ArrayBlockingQueue__E: ...
    def put(self, e: _ArrayBlockingQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _ArrayBlockingQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_ArrayBlockingQueue__E]
        | typing.Callable[[_ArrayBlockingQueue__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_ArrayBlockingQueue__E]: ...
    def take(self) -> _ArrayBlockingQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_BlockingDeque__E = typing.TypeVar("_BlockingDeque__E")  # <E>

class BlockingDeque(
    BlockingQueue[_BlockingDeque__E],
    java.util.Deque[_BlockingDeque__E],
    typing.Generic[_BlockingDeque__E],
):
    def add(self, e: _BlockingDeque__E) -> bool: ...
    def addFirst(self, e: _BlockingDeque__E) -> None: ...
    def addLast(self, e: _BlockingDeque__E) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def element(self) -> _BlockingDeque__E: ...
    def equals(self, object: typing.Any) -> bool: ...
    def hashCode(self) -> int: ...
    def iterator(self) -> java.util.Iterator[_BlockingDeque__E]: ...
    @typing.overload
    def offer(self, e: _BlockingDeque__E) -> bool: ...
    @typing.overload
    def offer(self, e: _BlockingDeque__E, long: int, timeUnit: TimeUnit) -> bool: ...
    @typing.overload
    def offerFirst(self, e: _BlockingDeque__E) -> bool: ...
    @typing.overload
    def offerFirst(
        self, e: _BlockingDeque__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    @typing.overload
    def offerLast(self, e: _BlockingDeque__E) -> bool: ...
    @typing.overload
    def offerLast(
        self, e: _BlockingDeque__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _BlockingDeque__E: ...
    @typing.overload
    def poll(self) -> _BlockingDeque__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _BlockingDeque__E: ...
    @typing.overload
    def pollFirst(self) -> _BlockingDeque__E: ...
    @typing.overload
    def pollFirst(self, long: int, timeUnit: TimeUnit) -> _BlockingDeque__E: ...
    @typing.overload
    def pollLast(self) -> _BlockingDeque__E: ...
    @typing.overload
    def pollLast(self, long: int, timeUnit: TimeUnit) -> _BlockingDeque__E: ...
    def push(self, e: _BlockingDeque__E) -> None: ...
    def put(self, e: _BlockingDeque__E) -> None: ...
    def putFirst(self, e: _BlockingDeque__E) -> None: ...
    def putLast(self, e: _BlockingDeque__E) -> None: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _BlockingDeque__E: ...
    def removeFirstOccurrence(self, object: typing.Any) -> bool: ...
    def removeLastOccurrence(self, object: typing.Any) -> bool: ...
    def size(self) -> int: ...
    def take(self) -> _BlockingDeque__E: ...
    def takeFirst(self) -> _BlockingDeque__E: ...
    def takeLast(self) -> _BlockingDeque__E: ...

_CompletableFuture__T = typing.TypeVar("_CompletableFuture__T")  # <T>

class CompletableFuture(
    Future[_CompletableFuture__T],
    CompletionStage[_CompletableFuture__T],
    typing.Generic[_CompletableFuture__T],
):
    def __init__(self): ...
    def acceptEither(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def acceptEitherAsync(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def acceptEitherAsync(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @staticmethod
    def allOf(
        *completableFuture: CompletableFuture[typing.Any],
    ) -> CompletableFuture[None]: ...
    @staticmethod
    def anyOf(
        *completableFuture: CompletableFuture[typing.Any],
    ) -> CompletableFuture[typing.Any]: ...
    _applyToEither__U = typing.TypeVar("_applyToEither__U")  # <U>
    def applyToEither(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        function: java.util.function.Function[_CompletableFuture__T, _applyToEither__U]
        | typing.Callable[[_CompletableFuture__T], _applyToEither__U],
    ) -> CompletableFuture[_applyToEither__U]: ...
    _applyToEitherAsync_0__U = typing.TypeVar("_applyToEitherAsync_0__U")  # <U>
    _applyToEitherAsync_1__U = typing.TypeVar("_applyToEitherAsync_1__U")  # <U>
    @typing.overload
    def applyToEitherAsync(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        function: java.util.function.Function[
            _CompletableFuture__T, _applyToEitherAsync_0__U
        ]
        | typing.Callable[[_CompletableFuture__T], _applyToEitherAsync_0__U],
    ) -> CompletableFuture[_applyToEitherAsync_0__U]: ...
    @typing.overload
    def applyToEitherAsync(
        self,
        completionStage: CompletionStage[_CompletableFuture__T],
        function: java.util.function.Function[
            _CompletableFuture__T, _applyToEitherAsync_1__U
        ]
        | typing.Callable[[_CompletableFuture__T], _applyToEitherAsync_1__U],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_applyToEitherAsync_1__U]: ...
    def cancel(self, boolean: bool) -> bool: ...
    def complete(self, t: _CompletableFuture__T) -> bool: ...
    @typing.overload
    def completeAsync(
        self,
        supplier: java.util.function.Supplier[_CompletableFuture__T]
        | typing.Callable[[], _CompletableFuture__T],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def completeAsync(
        self,
        supplier: java.util.function.Supplier[_CompletableFuture__T]
        | typing.Callable[[], _CompletableFuture__T],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    def completeExceptionally(self, throwable: java.lang.Throwable) -> bool: ...
    def completeOnTimeout(
        self, t: _CompletableFuture__T, long: int, timeUnit: TimeUnit
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    _completedFuture__U = typing.TypeVar("_completedFuture__U")  # <U>
    @staticmethod
    def completedFuture(
        u: _completedFuture__U,
    ) -> CompletableFuture[_completedFuture__U]: ...
    _completedStage__U = typing.TypeVar("_completedStage__U")  # <U>
    @staticmethod
    def completedStage(
        u: _completedStage__U,
    ) -> CompletionStage[_completedStage__U]: ...
    def copy(self) -> CompletableFuture[_CompletableFuture__T]: ...
    def defaultExecutor(self) -> Executor: ...
    @typing.overload
    @staticmethod
    def delayedExecutor(long: int, timeUnit: TimeUnit) -> Executor: ...
    @typing.overload
    @staticmethod
    def delayedExecutor(
        long: int, timeUnit: TimeUnit, executor: Executor | typing.Callable
    ) -> Executor: ...
    def exceptionNow(self) -> java.lang.Throwable: ...
    def exceptionally(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, _CompletableFuture__T
        ]
        | typing.Callable[[java.lang.Throwable], _CompletableFuture__T],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def exceptionallyAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, _CompletableFuture__T
        ]
        | typing.Callable[[java.lang.Throwable], _CompletableFuture__T],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def exceptionallyAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, _CompletableFuture__T
        ]
        | typing.Callable[[java.lang.Throwable], _CompletableFuture__T],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    def exceptionallyCompose(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletableFuture__T]
        ]
        | typing.Callable[
            [java.lang.Throwable], CompletionStage[_CompletableFuture__T]
        ],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def exceptionallyComposeAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletableFuture__T]
        ]
        | typing.Callable[
            [java.lang.Throwable], CompletionStage[_CompletableFuture__T]
        ],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def exceptionallyComposeAsync(
        self,
        function: java.util.function.Function[
            java.lang.Throwable, CompletionStage[_CompletableFuture__T]
        ]
        | typing.Callable[
            [java.lang.Throwable], CompletionStage[_CompletableFuture__T]
        ],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    _failedFuture__U = typing.TypeVar("_failedFuture__U")  # <U>
    @staticmethod
    def failedFuture(
        throwable: java.lang.Throwable,
    ) -> CompletableFuture[_failedFuture__U]: ...
    _failedStage__U = typing.TypeVar("_failedStage__U")  # <U>
    @staticmethod
    def failedStage(
        throwable: java.lang.Throwable,
    ) -> CompletionStage[_failedStage__U]: ...
    @typing.overload
    def get(self) -> _CompletableFuture__T: ...
    @typing.overload
    def get(self, long: int, timeUnit: TimeUnit) -> _CompletableFuture__T: ...
    def getNow(self, t: _CompletableFuture__T) -> _CompletableFuture__T: ...
    def getNumberOfDependents(self) -> int: ...
    _handle__U = typing.TypeVar("_handle__U")  # <U>
    def handle(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, java.lang.Throwable, _handle__U
        ]
        | typing.Callable[[_CompletableFuture__T, java.lang.Throwable], _handle__U],
    ) -> CompletableFuture[_handle__U]: ...
    _handleAsync_0__U = typing.TypeVar("_handleAsync_0__U")  # <U>
    _handleAsync_1__U = typing.TypeVar("_handleAsync_1__U")  # <U>
    @typing.overload
    def handleAsync(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, java.lang.Throwable, _handleAsync_0__U
        ]
        | typing.Callable[
            [_CompletableFuture__T, java.lang.Throwable], _handleAsync_0__U
        ],
    ) -> CompletableFuture[_handleAsync_0__U]: ...
    @typing.overload
    def handleAsync(
        self,
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, java.lang.Throwable, _handleAsync_1__U
        ]
        | typing.Callable[
            [_CompletableFuture__T, java.lang.Throwable], _handleAsync_1__U
        ],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_handleAsync_1__U]: ...
    def isCancelled(self) -> bool: ...
    def isCompletedExceptionally(self) -> bool: ...
    def isDone(self) -> bool: ...
    def join(self) -> _CompletableFuture__T: ...
    def minimalCompletionStage(self) -> CompletionStage[_CompletableFuture__T]: ...
    _newIncompleteFuture__U = typing.TypeVar("_newIncompleteFuture__U")  # <U>
    def newIncompleteFuture(self) -> CompletableFuture[_newIncompleteFuture__U]: ...
    def obtrudeException(self, throwable: java.lang.Throwable) -> None: ...
    def obtrudeValue(self, t: _CompletableFuture__T) -> None: ...
    def orTimeout(
        self, long: int, timeUnit: TimeUnit
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    def resultNow(self) -> _CompletableFuture__T: ...
    def runAfterBoth(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def runAfterBothAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def runAfterBothAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    def runAfterEither(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def runAfterEitherAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def runAfterEitherAsync(
        self,
        completionStage: CompletionStage[typing.Any],
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    @staticmethod
    def runAsync(
        runnable: java.lang.Runnable | typing.Callable,
    ) -> CompletableFuture[None]: ...
    @typing.overload
    @staticmethod
    def runAsync(
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    def state(self) -> Future.State: ...
    _supplyAsync_0__U = typing.TypeVar("_supplyAsync_0__U")  # <U>
    _supplyAsync_1__U = typing.TypeVar("_supplyAsync_1__U")  # <U>
    @typing.overload
    @staticmethod
    def supplyAsync(
        supplier: java.util.function.Supplier[_supplyAsync_0__U]
        | typing.Callable[[], _supplyAsync_0__U],
    ) -> CompletableFuture[_supplyAsync_0__U]: ...
    @typing.overload
    @staticmethod
    def supplyAsync(
        supplier: java.util.function.Supplier[_supplyAsync_1__U]
        | typing.Callable[[], _supplyAsync_1__U],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_supplyAsync_1__U]: ...
    def thenAccept(
        self,
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def thenAcceptAsync(
        self,
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def thenAcceptAsync(
        self,
        consumer: java.util.function.Consumer[_CompletableFuture__T]
        | typing.Callable[[_CompletableFuture__T], None],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    _thenAcceptBoth__U = typing.TypeVar("_thenAcceptBoth__U")  # <U>
    def thenAcceptBoth(
        self,
        completionStage: CompletionStage[_thenAcceptBoth__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, _thenAcceptBoth__U
        ]
        | typing.Callable[[_CompletableFuture__T, _thenAcceptBoth__U], None],
    ) -> CompletableFuture[None]: ...
    _thenAcceptBothAsync_0__U = typing.TypeVar("_thenAcceptBothAsync_0__U")  # <U>
    _thenAcceptBothAsync_1__U = typing.TypeVar("_thenAcceptBothAsync_1__U")  # <U>
    @typing.overload
    def thenAcceptBothAsync(
        self,
        completionStage: CompletionStage[_thenAcceptBothAsync_0__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, _thenAcceptBothAsync_0__U
        ]
        | typing.Callable[[_CompletableFuture__T, _thenAcceptBothAsync_0__U], None],
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def thenAcceptBothAsync(
        self,
        completionStage: CompletionStage[_thenAcceptBothAsync_1__U],
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, _thenAcceptBothAsync_1__U
        ]
        | typing.Callable[[_CompletableFuture__T, _thenAcceptBothAsync_1__U], None],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    _thenApply__U = typing.TypeVar("_thenApply__U")  # <U>
    def thenApply(
        self,
        function: java.util.function.Function[_CompletableFuture__T, _thenApply__U]
        | typing.Callable[[_CompletableFuture__T], _thenApply__U],
    ) -> CompletableFuture[_thenApply__U]: ...
    _thenApplyAsync_0__U = typing.TypeVar("_thenApplyAsync_0__U")  # <U>
    _thenApplyAsync_1__U = typing.TypeVar("_thenApplyAsync_1__U")  # <U>
    @typing.overload
    def thenApplyAsync(
        self,
        function: java.util.function.Function[
            _CompletableFuture__T, _thenApplyAsync_0__U
        ]
        | typing.Callable[[_CompletableFuture__T], _thenApplyAsync_0__U],
    ) -> CompletableFuture[_thenApplyAsync_0__U]: ...
    @typing.overload
    def thenApplyAsync(
        self,
        function: java.util.function.Function[
            _CompletableFuture__T, _thenApplyAsync_1__U
        ]
        | typing.Callable[[_CompletableFuture__T], _thenApplyAsync_1__U],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_thenApplyAsync_1__U]: ...
    _thenCombine__U = typing.TypeVar("_thenCombine__U")  # <U>
    _thenCombine__V = typing.TypeVar("_thenCombine__V")  # <V>
    def thenCombine(
        self,
        completionStage: CompletionStage[_thenCombine__U],
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, _thenCombine__U, _thenCombine__V
        ]
        | typing.Callable[[_CompletableFuture__T, _thenCombine__U], _thenCombine__V],
    ) -> CompletableFuture[_thenCombine__V]: ...
    _thenCombineAsync_0__U = typing.TypeVar("_thenCombineAsync_0__U")  # <U>
    _thenCombineAsync_0__V = typing.TypeVar("_thenCombineAsync_0__V")  # <V>
    _thenCombineAsync_1__U = typing.TypeVar("_thenCombineAsync_1__U")  # <U>
    _thenCombineAsync_1__V = typing.TypeVar("_thenCombineAsync_1__V")  # <V>
    @typing.overload
    def thenCombineAsync(
        self,
        completionStage: CompletionStage[_thenCombineAsync_0__U],
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, _thenCombineAsync_0__U, _thenCombineAsync_0__V
        ]
        | typing.Callable[
            [_CompletableFuture__T, _thenCombineAsync_0__U], _thenCombineAsync_0__V
        ],
    ) -> CompletableFuture[_thenCombineAsync_0__V]: ...
    @typing.overload
    def thenCombineAsync(
        self,
        completionStage: CompletionStage[_thenCombineAsync_1__U],
        biFunction: java.util.function.BiFunction[
            _CompletableFuture__T, _thenCombineAsync_1__U, _thenCombineAsync_1__V
        ]
        | typing.Callable[
            [_CompletableFuture__T, _thenCombineAsync_1__U], _thenCombineAsync_1__V
        ],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_thenCombineAsync_1__V]: ...
    _thenCompose__U = typing.TypeVar("_thenCompose__U")  # <U>
    def thenCompose(
        self,
        function: java.util.function.Function[
            _CompletableFuture__T, CompletionStage[_thenCompose__U]
        ]
        | typing.Callable[[_CompletableFuture__T], CompletionStage[_thenCompose__U]],
    ) -> CompletableFuture[_thenCompose__U]: ...
    _thenComposeAsync_0__U = typing.TypeVar("_thenComposeAsync_0__U")  # <U>
    _thenComposeAsync_1__U = typing.TypeVar("_thenComposeAsync_1__U")  # <U>
    @typing.overload
    def thenComposeAsync(
        self,
        function: java.util.function.Function[
            _CompletableFuture__T, CompletionStage[_thenComposeAsync_0__U]
        ]
        | typing.Callable[
            [_CompletableFuture__T], CompletionStage[_thenComposeAsync_0__U]
        ],
    ) -> CompletableFuture[_thenComposeAsync_0__U]: ...
    @typing.overload
    def thenComposeAsync(
        self,
        function: java.util.function.Function[
            _CompletableFuture__T, CompletionStage[_thenComposeAsync_1__U]
        ]
        | typing.Callable[
            [_CompletableFuture__T], CompletionStage[_thenComposeAsync_1__U]
        ],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_thenComposeAsync_1__U]: ...
    def thenRun(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def thenRunAsync(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> CompletableFuture[None]: ...
    @typing.overload
    def thenRunAsync(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[None]: ...
    def toCompletableFuture(self) -> CompletableFuture[_CompletableFuture__T]: ...
    def toString(self) -> java.lang.String: ...
    def whenComplete(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletableFuture__T, java.lang.Throwable], None],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def whenCompleteAsync(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletableFuture__T, java.lang.Throwable], None],
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    @typing.overload
    def whenCompleteAsync(
        self,
        biConsumer: java.util.function.BiConsumer[
            _CompletableFuture__T, java.lang.Throwable
        ]
        | typing.Callable[[_CompletableFuture__T, java.lang.Throwable], None],
        executor: Executor | typing.Callable,
    ) -> CompletableFuture[_CompletableFuture__T]: ...
    class AsynchronousCompletionTask: ...

_ConcurrentNavigableMap__K = typing.TypeVar("_ConcurrentNavigableMap__K")  # <K>
_ConcurrentNavigableMap__V = typing.TypeVar("_ConcurrentNavigableMap__V")  # <V>

class ConcurrentNavigableMap(
    ConcurrentMap[_ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V],
    java.util.NavigableMap[_ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V],
    typing.Generic[_ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V],
):
    def descendingKeySet(
        self,
    ) -> java.util.NavigableSet[_ConcurrentNavigableMap__K]: ...
    def descendingMap(
        self,
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    def equals(self, object: typing.Any) -> bool: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def headMap(
        self, k: _ConcurrentNavigableMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    @typing.overload
    def headMap(
        self, k: _ConcurrentNavigableMap__K, boolean: bool
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    def keySet(self) -> java.util.NavigableSet[_ConcurrentNavigableMap__K]: ...
    def navigableKeySet(self) -> java.util.NavigableSet[_ConcurrentNavigableMap__K]: ...
    @typing.overload
    def subMap(
        self,
        k: _ConcurrentNavigableMap__K,
        boolean: bool,
        k2: _ConcurrentNavigableMap__K,
        boolean2: bool,
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    @typing.overload
    def subMap(
        self, k: _ConcurrentNavigableMap__K, k2: _ConcurrentNavigableMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    @typing.overload
    def tailMap(
        self, k: _ConcurrentNavigableMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...
    @typing.overload
    def tailMap(
        self, k: _ConcurrentNavigableMap__K, boolean: bool
    ) -> ConcurrentNavigableMap[
        _ConcurrentNavigableMap__K, _ConcurrentNavigableMap__V
    ]: ...

_DelayQueue__E = typing.TypeVar("_DelayQueue__E", bound=Delayed)  # <E>

class DelayQueue(
    java.util.AbstractQueue[_DelayQueue__E],
    BlockingQueue[_DelayQueue__E],
    typing.Generic[_DelayQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_DelayQueue__E]
        | typing.Sequence[_DelayQueue__E]
        | set[_DelayQueue__E],
    ): ...
    def add(self, e: _DelayQueue__E) -> bool: ...
    def clear(self) -> None: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_DelayQueue__E]
        | typing.Sequence[_DelayQueue__E]
        | set[_DelayQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_DelayQueue__E]
        | typing.Sequence[_DelayQueue__E]
        | set[_DelayQueue__E],
        int: int,
    ) -> int: ...
    def iterator(self) -> java.util.Iterator[_DelayQueue__E]: ...
    @typing.overload
    def offer(self, e: _DelayQueue__E) -> bool: ...
    @typing.overload
    def offer(self, e: _DelayQueue__E, long: int, timeUnit: TimeUnit) -> bool: ...
    def peek(self) -> _DelayQueue__E: ...
    @typing.overload
    def poll(self) -> _DelayQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _DelayQueue__E: ...
    def put(self, e: _DelayQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _DelayQueue__E: ...
    def size(self) -> int: ...
    def take(self) -> _DelayQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...

_ExecutorCompletionService__V = typing.TypeVar("_ExecutorCompletionService__V")  # <V>

class ExecutorCompletionService(
    CompletionService[_ExecutorCompletionService__V],
    typing.Generic[_ExecutorCompletionService__V],
):
    @typing.overload
    def __init__(self, executor: Executor | typing.Callable): ...
    @typing.overload
    def __init__(
        self,
        executor: Executor | typing.Callable,
        blockingQueue: BlockingQueue[Future[_ExecutorCompletionService__V]],
    ): ...
    @typing.overload
    def poll(self) -> Future[_ExecutorCompletionService__V]: ...
    @typing.overload
    def poll(
        self, long: int, timeUnit: TimeUnit
    ) -> Future[_ExecutorCompletionService__V]: ...
    @typing.overload
    def submit(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        v: _ExecutorCompletionService__V,
    ) -> Future[_ExecutorCompletionService__V]: ...
    @typing.overload
    def submit(
        self,
        callable: Callable[_ExecutorCompletionService__V]
        | typing.Callable[[], _ExecutorCompletionService__V],
    ) -> Future[_ExecutorCompletionService__V]: ...
    def take(self) -> Future[_ExecutorCompletionService__V]: ...

class ExecutorService(Executor, java.lang.AutoCloseable):
    def awaitTermination(self, long: int, timeUnit: TimeUnit) -> bool: ...
    def close(self) -> None: ...
    _invokeAll_0__T = typing.TypeVar("_invokeAll_0__T")  # <T>
    _invokeAll_1__T = typing.TypeVar("_invokeAll_1__T")  # <T>
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_0__T]]
        | typing.Sequence[Callable[_invokeAll_0__T]]
        | set[Callable[_invokeAll_0__T]],
    ) -> java.util.List[Future[_invokeAll_0__T]]: ...
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_1__T]]
        | typing.Sequence[Callable[_invokeAll_1__T]]
        | set[Callable[_invokeAll_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> java.util.List[Future[_invokeAll_1__T]]: ...
    _invokeAny_0__T = typing.TypeVar("_invokeAny_0__T")  # <T>
    _invokeAny_1__T = typing.TypeVar("_invokeAny_1__T")  # <T>
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_0__T]]
        | typing.Sequence[Callable[_invokeAny_0__T]]
        | set[Callable[_invokeAny_0__T]],
    ) -> _invokeAny_0__T: ...
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_1__T]]
        | typing.Sequence[Callable[_invokeAny_1__T]]
        | set[Callable[_invokeAny_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> _invokeAny_1__T: ...
    def isShutdown(self) -> bool: ...
    def isTerminated(self) -> bool: ...
    def shutdown(self) -> None: ...
    def shutdownNow(self) -> java.util.List[java.lang.Runnable]: ...
    _submit_1__T = typing.TypeVar("_submit_1__T")  # <T>
    _submit_2__T = typing.TypeVar("_submit_2__T")  # <T>
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> Future[typing.Any]: ...
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable, t: _submit_1__T
    ) -> Future[_submit_1__T]: ...
    @typing.overload
    def submit(
        self, callable: Callable[_submit_2__T] | typing.Callable[[], _submit_2__T]
    ) -> Future[_submit_2__T]: ...

_ForkJoinTask__V = typing.TypeVar("_ForkJoinTask__V")  # <V>

class ForkJoinTask(
    Future[_ForkJoinTask__V], java.io.Serializable, typing.Generic[_ForkJoinTask__V]
):
    def __init__(self): ...
    _adapt_1__T = typing.TypeVar("_adapt_1__T")  # <T>
    _adapt_2__T = typing.TypeVar("_adapt_2__T")  # <T>
    @typing.overload
    @staticmethod
    def adapt(
        runnable: java.lang.Runnable | typing.Callable,
    ) -> ForkJoinTask[typing.Any]: ...
    @typing.overload
    @staticmethod
    def adapt(
        runnable: java.lang.Runnable | typing.Callable, t: _adapt_1__T
    ) -> ForkJoinTask[_adapt_1__T]: ...
    @typing.overload
    @staticmethod
    def adapt(
        callable: Callable[_adapt_2__T] | typing.Callable[[], _adapt_2__T],
    ) -> ForkJoinTask[_adapt_2__T]: ...
    _adaptInterruptible_1__T = typing.TypeVar("_adaptInterruptible_1__T")  # <T>
    _adaptInterruptible_2__T = typing.TypeVar("_adaptInterruptible_2__T")  # <T>
    @typing.overload
    @staticmethod
    def adaptInterruptible(
        runnable: java.lang.Runnable | typing.Callable,
    ) -> ForkJoinTask[typing.Any]: ...
    @typing.overload
    @staticmethod
    def adaptInterruptible(
        runnable: java.lang.Runnable | typing.Callable, t: _adaptInterruptible_1__T
    ) -> ForkJoinTask[_adaptInterruptible_1__T]: ...
    @typing.overload
    @staticmethod
    def adaptInterruptible(
        callable: Callable[_adaptInterruptible_2__T]
        | typing.Callable[[], _adaptInterruptible_2__T],
    ) -> ForkJoinTask[_adaptInterruptible_2__T]: ...
    def cancel(self, boolean: bool) -> bool: ...
    def compareAndSetForkJoinTaskTag(self, short: int, short2: int) -> bool: ...
    def complete(self, v: _ForkJoinTask__V) -> None: ...
    def completeExceptionally(self, throwable: java.lang.Throwable) -> None: ...
    def exceptionNow(self) -> java.lang.Throwable: ...
    def fork(self) -> ForkJoinTask[_ForkJoinTask__V]: ...
    @typing.overload
    def get(self) -> _ForkJoinTask__V: ...
    @typing.overload
    def get(self, long: int, timeUnit: TimeUnit) -> _ForkJoinTask__V: ...
    def getException(self) -> java.lang.Throwable: ...
    def getForkJoinTaskTag(self) -> int: ...
    @staticmethod
    def getPool() -> ForkJoinPool: ...
    @staticmethod
    def getQueuedTaskCount() -> int: ...
    def getRawResult(self) -> _ForkJoinTask__V: ...
    @staticmethod
    def getSurplusQueuedTaskCount() -> int: ...
    @staticmethod
    def helpQuiesce() -> None: ...
    @staticmethod
    def inForkJoinPool() -> bool: ...
    def invoke(self) -> _ForkJoinTask__V: ...
    _invokeAll_0__T = typing.TypeVar("_invokeAll_0__T", bound="ForkJoinTask")  # <T>
    @typing.overload
    @staticmethod
    def invokeAll(
        collection: java.util.Collection[_invokeAll_0__T]
        | typing.Sequence[_invokeAll_0__T]
        | set[_invokeAll_0__T],
    ) -> java.util.Collection[_invokeAll_0__T]: ...
    @typing.overload
    @staticmethod
    def invokeAll(
        forkJoinTask: ForkJoinTask[typing.Any], forkJoinTask2: ForkJoinTask[typing.Any]
    ) -> None: ...
    @typing.overload
    @staticmethod
    def invokeAll(*forkJoinTask: ForkJoinTask[typing.Any]) -> None: ...
    def isCancelled(self) -> bool: ...
    def isCompletedAbnormally(self) -> bool: ...
    def isCompletedNormally(self) -> bool: ...
    def isDone(self) -> bool: ...
    def join(self) -> _ForkJoinTask__V: ...
    def quietlyComplete(self) -> None: ...
    def quietlyInvoke(self) -> None: ...
    @typing.overload
    def quietlyJoin(self, long: int, timeUnit: TimeUnit) -> bool: ...
    @typing.overload
    def quietlyJoin(self) -> None: ...
    def quietlyJoinUninterruptibly(self, long: int, timeUnit: TimeUnit) -> bool: ...
    def reinitialize(self) -> None: ...
    def resultNow(self) -> _ForkJoinTask__V: ...
    def setForkJoinTaskTag(self, short: int) -> int: ...
    def state(self) -> Future.State: ...
    def tryUnfork(self) -> bool: ...

_LinkedBlockingQueue__E = typing.TypeVar("_LinkedBlockingQueue__E")  # <E>

class LinkedBlockingQueue(
    java.util.AbstractQueue[_LinkedBlockingQueue__E],
    BlockingQueue[_LinkedBlockingQueue__E],
    java.io.Serializable,
    typing.Generic[_LinkedBlockingQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_LinkedBlockingQueue__E]
        | typing.Sequence[_LinkedBlockingQueue__E]
        | set[_LinkedBlockingQueue__E],
    ): ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedBlockingQueue__E]
        | typing.Sequence[_LinkedBlockingQueue__E]
        | set[_LinkedBlockingQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedBlockingQueue__E]
        | typing.Sequence[_LinkedBlockingQueue__E]
        | set[_LinkedBlockingQueue__E],
        int: int,
    ) -> int: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_LinkedBlockingQueue__E]
        | typing.Callable[[_LinkedBlockingQueue__E], None],
    ) -> None: ...
    def iterator(self) -> java.util.Iterator[_LinkedBlockingQueue__E]: ...
    @typing.overload
    def offer(self, e: _LinkedBlockingQueue__E) -> bool: ...
    @typing.overload
    def offer(
        self, e: _LinkedBlockingQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _LinkedBlockingQueue__E: ...
    @typing.overload
    def poll(self) -> _LinkedBlockingQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _LinkedBlockingQueue__E: ...
    def put(self, e: _LinkedBlockingQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _LinkedBlockingQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_LinkedBlockingQueue__E]
        | typing.Callable[[_LinkedBlockingQueue__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_LinkedBlockingQueue__E]: ...
    def take(self) -> _LinkedBlockingQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_PriorityBlockingQueue__E = typing.TypeVar("_PriorityBlockingQueue__E")  # <E>

class PriorityBlockingQueue(
    java.util.AbstractQueue[_PriorityBlockingQueue__E],
    BlockingQueue[_PriorityBlockingQueue__E],
    java.io.Serializable,
    typing.Generic[_PriorityBlockingQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        comparator: java.util.Comparator[_PriorityBlockingQueue__E]
        | typing.Callable[[_PriorityBlockingQueue__E, _PriorityBlockingQueue__E], int],
    ): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_PriorityBlockingQueue__E]
        | typing.Sequence[_PriorityBlockingQueue__E]
        | set[_PriorityBlockingQueue__E],
    ): ...
    def add(self, e: _PriorityBlockingQueue__E) -> bool: ...
    def clear(self) -> None: ...
    def comparator(self) -> java.util.Comparator[_PriorityBlockingQueue__E]: ...
    def contains(self, object: typing.Any) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_PriorityBlockingQueue__E]
        | typing.Sequence[_PriorityBlockingQueue__E]
        | set[_PriorityBlockingQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_PriorityBlockingQueue__E]
        | typing.Sequence[_PriorityBlockingQueue__E]
        | set[_PriorityBlockingQueue__E],
        int: int,
    ) -> int: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_PriorityBlockingQueue__E]
        | typing.Callable[[_PriorityBlockingQueue__E], None],
    ) -> None: ...
    def iterator(self) -> java.util.Iterator[_PriorityBlockingQueue__E]: ...
    @typing.overload
    def offer(self, e: _PriorityBlockingQueue__E) -> bool: ...
    @typing.overload
    def offer(
        self, e: _PriorityBlockingQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _PriorityBlockingQueue__E: ...
    @typing.overload
    def poll(self) -> _PriorityBlockingQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _PriorityBlockingQueue__E: ...
    def put(self, e: _PriorityBlockingQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _PriorityBlockingQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_PriorityBlockingQueue__E]
        | typing.Callable[[_PriorityBlockingQueue__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_PriorityBlockingQueue__E]: ...
    def take(self) -> _PriorityBlockingQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_RunnableFuture__V = typing.TypeVar("_RunnableFuture__V")  # <V>

class RunnableFuture(
    java.lang.Runnable, Future[_RunnableFuture__V], typing.Generic[_RunnableFuture__V]
):
    def run(self) -> None: ...

_ScheduledFuture__V = typing.TypeVar("_ScheduledFuture__V")  # <V>

class ScheduledFuture(
    Delayed, Future[_ScheduledFuture__V], typing.Generic[_ScheduledFuture__V]
): ...

_SynchronousQueue__E = typing.TypeVar("_SynchronousQueue__E")  # <E>

class SynchronousQueue(
    java.util.AbstractQueue[_SynchronousQueue__E],
    BlockingQueue[_SynchronousQueue__E],
    java.io.Serializable,
    typing.Generic[_SynchronousQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, boolean: bool): ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def containsAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_SynchronousQueue__E]
        | typing.Sequence[_SynchronousQueue__E]
        | set[_SynchronousQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_SynchronousQueue__E]
        | typing.Sequence[_SynchronousQueue__E]
        | set[_SynchronousQueue__E],
        int: int,
    ) -> int: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_SynchronousQueue__E]: ...
    @typing.overload
    def offer(self, e: _SynchronousQueue__E) -> bool: ...
    @typing.overload
    def offer(self, e: _SynchronousQueue__E, long: int, timeUnit: TimeUnit) -> bool: ...
    def peek(self) -> _SynchronousQueue__E: ...
    @typing.overload
    def poll(self) -> _SynchronousQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _SynchronousQueue__E: ...
    def put(self, e: _SynchronousQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _SynchronousQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_SynchronousQueue__E]: ...
    def take(self) -> _SynchronousQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_TransferQueue__E = typing.TypeVar("_TransferQueue__E")  # <E>

class TransferQueue(
    BlockingQueue[_TransferQueue__E], typing.Generic[_TransferQueue__E]
):
    def equals(self, object: typing.Any) -> bool: ...
    def getWaitingConsumerCount(self) -> int: ...
    def hasWaitingConsumer(self) -> bool: ...
    def hashCode(self) -> int: ...
    def transfer(self, e: _TransferQueue__E) -> None: ...
    @typing.overload
    def tryTransfer(self, e: _TransferQueue__E) -> bool: ...
    @typing.overload
    def tryTransfer(
        self, e: _TransferQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...

class AbstractExecutorService(ExecutorService):
    def __init__(self): ...
    _invokeAll_0__T = typing.TypeVar("_invokeAll_0__T")  # <T>
    _invokeAll_1__T = typing.TypeVar("_invokeAll_1__T")  # <T>
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_0__T]]
        | typing.Sequence[Callable[_invokeAll_0__T]]
        | set[Callable[_invokeAll_0__T]],
    ) -> java.util.List[Future[_invokeAll_0__T]]: ...
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_1__T]]
        | typing.Sequence[Callable[_invokeAll_1__T]]
        | set[Callable[_invokeAll_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> java.util.List[Future[_invokeAll_1__T]]: ...
    _invokeAny_0__T = typing.TypeVar("_invokeAny_0__T")  # <T>
    _invokeAny_1__T = typing.TypeVar("_invokeAny_1__T")  # <T>
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_0__T]]
        | typing.Sequence[Callable[_invokeAny_0__T]]
        | set[Callable[_invokeAny_0__T]],
    ) -> _invokeAny_0__T: ...
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_1__T]]
        | typing.Sequence[Callable[_invokeAny_1__T]]
        | set[Callable[_invokeAny_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> _invokeAny_1__T: ...
    _submit_1__T = typing.TypeVar("_submit_1__T")  # <T>
    _submit_2__T = typing.TypeVar("_submit_2__T")  # <T>
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> Future[typing.Any]: ...
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable, t: _submit_1__T
    ) -> Future[_submit_1__T]: ...
    @typing.overload
    def submit(
        self, callable: Callable[_submit_2__T] | typing.Callable[[], _submit_2__T]
    ) -> Future[_submit_2__T]: ...

_ConcurrentSkipListMap__K = typing.TypeVar("_ConcurrentSkipListMap__K")  # <K>
_ConcurrentSkipListMap__V = typing.TypeVar("_ConcurrentSkipListMap__V")  # <V>

class ConcurrentSkipListMap(
    java.util.AbstractMap[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
    ConcurrentNavigableMap[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
    java.lang.Cloneable,
    java.io.Serializable,
    typing.Generic[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        comparator: java.util.Comparator[_ConcurrentSkipListMap__K]
        | typing.Callable[[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__K], int],
    ): ...
    @typing.overload
    def __init__(
        self,
        map: java.util.Map[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]
        | typing.Mapping[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
    ): ...
    @typing.overload
    def __init__(
        self,
        sortedMap: java.util.SortedMap[
            _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
        ],
    ): ...
    def ceilingEntry(
        self, k: _ConcurrentSkipListMap__K
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def ceilingKey(self, k: _ConcurrentSkipListMap__K) -> _ConcurrentSkipListMap__K: ...
    def clear(self) -> None: ...
    def clone(
        self,
    ) -> ConcurrentSkipListMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    def comparator(self) -> java.util.Comparator[_ConcurrentSkipListMap__K]: ...
    def compute(
        self,
        k: _ConcurrentSkipListMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentSkipListMap__K,
            _ConcurrentSkipListMap__V,
            _ConcurrentSkipListMap__V,
        ]
        | typing.Callable[
            [_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
            _ConcurrentSkipListMap__V,
        ],
    ) -> _ConcurrentSkipListMap__V: ...
    def computeIfAbsent(
        self,
        k: _ConcurrentSkipListMap__K,
        function: java.util.function.Function[
            _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
        ]
        | typing.Callable[[_ConcurrentSkipListMap__K], _ConcurrentSkipListMap__V],
    ) -> _ConcurrentSkipListMap__V: ...
    def computeIfPresent(
        self,
        k: _ConcurrentSkipListMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentSkipListMap__K,
            _ConcurrentSkipListMap__V,
            _ConcurrentSkipListMap__V,
        ]
        | typing.Callable[
            [_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
            _ConcurrentSkipListMap__V,
        ],
    ) -> _ConcurrentSkipListMap__V: ...
    def containsKey(self, object: typing.Any) -> bool: ...
    def containsValue(self, object: typing.Any) -> bool: ...
    def descendingKeySet(self) -> java.util.NavigableSet[_ConcurrentSkipListMap__K]: ...
    def descendingMap(
        self,
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    def entrySet(
        self,
    ) -> java.util.Set[
        java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]
    ]: ...
    def equals(self, object: typing.Any) -> bool: ...
    def firstEntry(
        self,
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def firstKey(self) -> _ConcurrentSkipListMap__K: ...
    def floorEntry(
        self, k: _ConcurrentSkipListMap__K
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def floorKey(self, k: _ConcurrentSkipListMap__K) -> _ConcurrentSkipListMap__K: ...
    def forEach(
        self,
        biConsumer: java.util.function.BiConsumer[
            _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
        ]
        | typing.Callable[[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V], None],
    ) -> None: ...
    def get(self, object: typing.Any) -> _ConcurrentSkipListMap__V: ...
    def getOrDefault(
        self, object: typing.Any, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    @typing.overload
    def headMap(
        self, k: _ConcurrentSkipListMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    @typing.overload
    def headMap(
        self, k: _ConcurrentSkipListMap__K, boolean: bool
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    def higherEntry(
        self, k: _ConcurrentSkipListMap__K
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def higherKey(self, k: _ConcurrentSkipListMap__K) -> _ConcurrentSkipListMap__K: ...
    def isEmpty(self) -> bool: ...
    def keySet(self) -> java.util.NavigableSet[_ConcurrentSkipListMap__K]: ...
    def lastEntry(
        self,
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def lastKey(self) -> _ConcurrentSkipListMap__K: ...
    def lowerEntry(
        self, k: _ConcurrentSkipListMap__K
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def lowerKey(self, k: _ConcurrentSkipListMap__K) -> _ConcurrentSkipListMap__K: ...
    def merge(
        self,
        k: _ConcurrentSkipListMap__K,
        v: _ConcurrentSkipListMap__V,
        biFunction: java.util.function.BiFunction[
            _ConcurrentSkipListMap__V,
            _ConcurrentSkipListMap__V,
            _ConcurrentSkipListMap__V,
        ]
        | typing.Callable[
            [_ConcurrentSkipListMap__V, _ConcurrentSkipListMap__V],
            _ConcurrentSkipListMap__V,
        ],
    ) -> _ConcurrentSkipListMap__V: ...
    def navigableKeySet(self) -> java.util.NavigableSet[_ConcurrentSkipListMap__K]: ...
    def pollFirstEntry(
        self,
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def pollLastEntry(
        self,
    ) -> java.util.Map.Entry[_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V]: ...
    def put(
        self, k: _ConcurrentSkipListMap__K, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    def putFirst(
        self, k: _ConcurrentSkipListMap__K, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    def putIfAbsent(
        self, k: _ConcurrentSkipListMap__K, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    def putLast(
        self, k: _ConcurrentSkipListMap__K, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    @typing.overload
    def remove(self, object: typing.Any, object2: typing.Any) -> bool: ...
    @typing.overload
    def remove(self, object: typing.Any) -> _ConcurrentSkipListMap__V: ...
    @typing.overload
    def replace(
        self,
        k: _ConcurrentSkipListMap__K,
        v: _ConcurrentSkipListMap__V,
        v2: _ConcurrentSkipListMap__V,
    ) -> bool: ...
    @typing.overload
    def replace(
        self, k: _ConcurrentSkipListMap__K, v: _ConcurrentSkipListMap__V
    ) -> _ConcurrentSkipListMap__V: ...
    def replaceAll(
        self,
        biFunction: java.util.function.BiFunction[
            _ConcurrentSkipListMap__K,
            _ConcurrentSkipListMap__V,
            _ConcurrentSkipListMap__V,
        ]
        | typing.Callable[
            [_ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V],
            _ConcurrentSkipListMap__V,
        ],
    ) -> None: ...
    def size(self) -> int: ...
    @typing.overload
    def subMap(
        self,
        k: _ConcurrentSkipListMap__K,
        boolean: bool,
        k2: _ConcurrentSkipListMap__K,
        boolean2: bool,
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    @typing.overload
    def subMap(
        self, k: _ConcurrentSkipListMap__K, k2: _ConcurrentSkipListMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    @typing.overload
    def tailMap(
        self, k: _ConcurrentSkipListMap__K
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    @typing.overload
    def tailMap(
        self, k: _ConcurrentSkipListMap__K, boolean: bool
    ) -> ConcurrentNavigableMap[
        _ConcurrentSkipListMap__K, _ConcurrentSkipListMap__V
    ]: ...
    def values(self) -> java.util.Collection[_ConcurrentSkipListMap__V]: ...

_CountedCompleter__T = typing.TypeVar("_CountedCompleter__T")  # <T>

class CountedCompleter(
    ForkJoinTask[_CountedCompleter__T], typing.Generic[_CountedCompleter__T]
):
    def addToPendingCount(self, int: int) -> None: ...
    def compareAndSetPendingCount(self, int: int, int2: int) -> bool: ...
    def complete(self, t: _CountedCompleter__T) -> None: ...
    def compute(self) -> None: ...
    def decrementPendingCountUnlessZero(self) -> int: ...
    def firstComplete(self) -> CountedCompleter[typing.Any]: ...
    def getCompleter(self) -> CountedCompleter[typing.Any]: ...
    def getPendingCount(self) -> int: ...
    def getRawResult(self) -> _CountedCompleter__T: ...
    def getRoot(self) -> CountedCompleter[typing.Any]: ...
    def helpComplete(self, int: int) -> None: ...
    def nextComplete(self) -> CountedCompleter[typing.Any]: ...
    def onCompletion(self, countedCompleter: CountedCompleter[typing.Any]) -> None: ...
    def onExceptionalCompletion(
        self,
        throwable: java.lang.Throwable,
        countedCompleter: CountedCompleter[typing.Any],
    ) -> bool: ...
    def propagateCompletion(self) -> None: ...
    def quietlyCompleteRoot(self) -> None: ...
    def setPendingCount(self, int: int) -> None: ...
    def tryComplete(self) -> None: ...

_FutureTask__V = typing.TypeVar("_FutureTask__V")  # <V>

class FutureTask(RunnableFuture[_FutureTask__V], typing.Generic[_FutureTask__V]):
    @typing.overload
    def __init__(
        self, runnable: java.lang.Runnable | typing.Callable, v: _FutureTask__V
    ): ...
    @typing.overload
    def __init__(
        self, callable: Callable[_FutureTask__V] | typing.Callable[[], _FutureTask__V]
    ): ...
    def cancel(self, boolean: bool) -> bool: ...
    def exceptionNow(self) -> java.lang.Throwable: ...
    @typing.overload
    def get(self) -> _FutureTask__V: ...
    @typing.overload
    def get(self, long: int, timeUnit: TimeUnit) -> _FutureTask__V: ...
    def isCancelled(self) -> bool: ...
    def isDone(self) -> bool: ...
    def resultNow(self) -> _FutureTask__V: ...
    def run(self) -> None: ...
    def state(self) -> Future.State: ...
    def toString(self) -> java.lang.String: ...

_LinkedBlockingDeque__E = typing.TypeVar("_LinkedBlockingDeque__E")  # <E>

class LinkedBlockingDeque(
    java.util.AbstractQueue[_LinkedBlockingDeque__E],
    BlockingDeque[_LinkedBlockingDeque__E],
    java.io.Serializable,
    typing.Generic[_LinkedBlockingDeque__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_LinkedBlockingDeque__E]
        | typing.Sequence[_LinkedBlockingDeque__E]
        | set[_LinkedBlockingDeque__E],
    ): ...
    def add(self, e: _LinkedBlockingDeque__E) -> bool: ...
    def addAll(
        self,
        collection: java.util.Collection[_LinkedBlockingDeque__E]
        | typing.Sequence[_LinkedBlockingDeque__E]
        | set[_LinkedBlockingDeque__E],
    ) -> bool: ...
    def addFirst(self, e: _LinkedBlockingDeque__E) -> None: ...
    def addLast(self, e: _LinkedBlockingDeque__E) -> None: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    def descendingIterator(self) -> java.util.Iterator[_LinkedBlockingDeque__E]: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedBlockingDeque__E]
        | typing.Sequence[_LinkedBlockingDeque__E]
        | set[_LinkedBlockingDeque__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedBlockingDeque__E]
        | typing.Sequence[_LinkedBlockingDeque__E]
        | set[_LinkedBlockingDeque__E],
        int: int,
    ) -> int: ...
    def element(self) -> _LinkedBlockingDeque__E: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_LinkedBlockingDeque__E]
        | typing.Callable[[_LinkedBlockingDeque__E], None],
    ) -> None: ...
    def getFirst(self) -> _LinkedBlockingDeque__E: ...
    def getLast(self) -> _LinkedBlockingDeque__E: ...
    def iterator(self) -> java.util.Iterator[_LinkedBlockingDeque__E]: ...
    @typing.overload
    def offer(self, e: _LinkedBlockingDeque__E) -> bool: ...
    @typing.overload
    def offer(
        self, e: _LinkedBlockingDeque__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    @typing.overload
    def offerFirst(self, e: _LinkedBlockingDeque__E) -> bool: ...
    @typing.overload
    def offerFirst(
        self, e: _LinkedBlockingDeque__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    @typing.overload
    def offerLast(self, e: _LinkedBlockingDeque__E) -> bool: ...
    @typing.overload
    def offerLast(
        self, e: _LinkedBlockingDeque__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _LinkedBlockingDeque__E: ...
    def peekFirst(self) -> _LinkedBlockingDeque__E: ...
    def peekLast(self) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def poll(self) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def pollFirst(self) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def pollFirst(self, long: int, timeUnit: TimeUnit) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def pollLast(self) -> _LinkedBlockingDeque__E: ...
    @typing.overload
    def pollLast(self, long: int, timeUnit: TimeUnit) -> _LinkedBlockingDeque__E: ...
    def pop(self) -> _LinkedBlockingDeque__E: ...
    def push(self, e: _LinkedBlockingDeque__E) -> None: ...
    def put(self, e: _LinkedBlockingDeque__E) -> None: ...
    def putFirst(self, e: _LinkedBlockingDeque__E) -> None: ...
    def putLast(self, e: _LinkedBlockingDeque__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _LinkedBlockingDeque__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeFirst(self) -> _LinkedBlockingDeque__E: ...
    def removeFirstOccurrence(self, object: typing.Any) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_LinkedBlockingDeque__E]
        | typing.Callable[[_LinkedBlockingDeque__E], bool],
    ) -> bool: ...
    def removeLast(self) -> _LinkedBlockingDeque__E: ...
    def removeLastOccurrence(self, object: typing.Any) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_LinkedBlockingDeque__E]: ...
    def take(self) -> _LinkedBlockingDeque__E: ...
    def takeFirst(self) -> _LinkedBlockingDeque__E: ...
    def takeLast(self) -> _LinkedBlockingDeque__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...

_LinkedTransferQueue__E = typing.TypeVar("_LinkedTransferQueue__E")  # <E>

class LinkedTransferQueue(
    java.util.AbstractQueue[_LinkedTransferQueue__E],
    TransferQueue[_LinkedTransferQueue__E],
    java.io.Serializable,
    typing.Generic[_LinkedTransferQueue__E],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        collection: java.util.Collection[_LinkedTransferQueue__E]
        | typing.Sequence[_LinkedTransferQueue__E]
        | set[_LinkedTransferQueue__E],
    ): ...
    def add(self, e: _LinkedTransferQueue__E) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, object: typing.Any) -> bool: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedTransferQueue__E]
        | typing.Sequence[_LinkedTransferQueue__E]
        | set[_LinkedTransferQueue__E],
    ) -> int: ...
    @typing.overload
    def drainTo(
        self,
        collection: java.util.Collection[_LinkedTransferQueue__E]
        | typing.Sequence[_LinkedTransferQueue__E]
        | set[_LinkedTransferQueue__E],
        int: int,
    ) -> int: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_LinkedTransferQueue__E]
        | typing.Callable[[_LinkedTransferQueue__E], None],
    ) -> None: ...
    def getWaitingConsumerCount(self) -> int: ...
    def hasWaitingConsumer(self) -> bool: ...
    def isEmpty(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_LinkedTransferQueue__E]: ...
    @typing.overload
    def offer(self, e: _LinkedTransferQueue__E) -> bool: ...
    @typing.overload
    def offer(
        self, e: _LinkedTransferQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...
    def peek(self) -> _LinkedTransferQueue__E: ...
    @typing.overload
    def poll(self) -> _LinkedTransferQueue__E: ...
    @typing.overload
    def poll(self, long: int, timeUnit: TimeUnit) -> _LinkedTransferQueue__E: ...
    def put(self, e: _LinkedTransferQueue__E) -> None: ...
    def remainingCapacity(self) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any) -> bool: ...
    @typing.overload
    def remove(self) -> _LinkedTransferQueue__E: ...
    def removeAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def removeIf(
        self,
        predicate: java.util.function.Predicate[_LinkedTransferQueue__E]
        | typing.Callable[[_LinkedTransferQueue__E], bool],
    ) -> bool: ...
    def retainAll(
        self,
        collection: java.util.Collection[typing.Any]
        | typing.Sequence[typing.Any]
        | set[typing.Any],
    ) -> bool: ...
    def size(self) -> int: ...
    def spliterator(self) -> java.util.Spliterator[_LinkedTransferQueue__E]: ...
    def take(self) -> _LinkedTransferQueue__E: ...
    _toArray_0__T = typing.TypeVar("_toArray_0__T")  # <T>
    _toArray_2__T = typing.TypeVar("_toArray_2__T")  # <T>
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_0__T] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_0__T] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_0__T]: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self, tArray: list[_toArray_2__T] | jpype.JArray
    ) -> typing.MutableSequence[_toArray_2__T]: ...
    def toString(self) -> java.lang.String: ...
    def transfer(self, e: _LinkedTransferQueue__E) -> None: ...
    @typing.overload
    def tryTransfer(self, e: _LinkedTransferQueue__E) -> bool: ...
    @typing.overload
    def tryTransfer(
        self, e: _LinkedTransferQueue__E, long: int, timeUnit: TimeUnit
    ) -> bool: ...

class RecursiveAction(ForkJoinTask[None]):
    def __init__(self): ...
    def getRawResult(self) -> None: ...

_RecursiveTask__V = typing.TypeVar("_RecursiveTask__V")  # <V>

class RecursiveTask(ForkJoinTask[_RecursiveTask__V], typing.Generic[_RecursiveTask__V]):
    def __init__(self): ...
    def getRawResult(self) -> _RecursiveTask__V: ...

_RunnableScheduledFuture__V = typing.TypeVar("_RunnableScheduledFuture__V")  # <V>

class RunnableScheduledFuture(
    RunnableFuture[_RunnableScheduledFuture__V],
    ScheduledFuture[_RunnableScheduledFuture__V],
    typing.Generic[_RunnableScheduledFuture__V],
):
    def isPeriodic(self) -> bool: ...

class ScheduledExecutorService(ExecutorService):
    _schedule_1__V = typing.TypeVar("_schedule_1__V")  # <V>
    @typing.overload
    def schedule(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    @typing.overload
    def schedule(
        self,
        callable: Callable[_schedule_1__V] | typing.Callable[[], _schedule_1__V],
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[_schedule_1__V]: ...
    def scheduleAtFixedRate(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    def scheduleWithFixedDelay(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...

class ForkJoinPool(AbstractExecutorService, ScheduledExecutorService):
    defaultForkJoinWorkerThreadFactory: typing.ClassVar[
        ForkJoinPool.ForkJoinWorkerThreadFactory
    ] = ...
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        forkJoinWorkerThreadFactory: ForkJoinPool.ForkJoinWorkerThreadFactory
        | typing.Callable,
        uncaughtExceptionHandler: java.lang.Thread.UncaughtExceptionHandler
        | typing.Callable,
        boolean: bool,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        forkJoinWorkerThreadFactory: ForkJoinPool.ForkJoinWorkerThreadFactory
        | typing.Callable,
        uncaughtExceptionHandler: java.lang.Thread.UncaughtExceptionHandler
        | typing.Callable,
        boolean: bool,
        int2: int,
        int3: int,
        int4: int,
        predicate: java.util.function.Predicate[ForkJoinPool]
        | typing.Callable[[ForkJoinPool], bool],
        long: int,
        timeUnit: TimeUnit,
    ): ...
    def awaitQuiescence(self, long: int, timeUnit: TimeUnit) -> bool: ...
    def awaitTermination(self, long: int, timeUnit: TimeUnit) -> bool: ...
    def cancelDelayedTasksOnShutdown(self) -> None: ...
    def close(self) -> None: ...
    @staticmethod
    def commonPool() -> ForkJoinPool: ...
    @typing.overload
    def execute(self, runnable: java.lang.Runnable | typing.Callable) -> None: ...
    @typing.overload
    def execute(self, forkJoinTask: ForkJoinTask[typing.Any]) -> None: ...
    _externalSubmit__T = typing.TypeVar("_externalSubmit__T")  # <T>
    def externalSubmit(
        self, forkJoinTask: ForkJoinTask[_externalSubmit__T]
    ) -> ForkJoinTask[_externalSubmit__T]: ...
    def getActiveThreadCount(self) -> int: ...
    def getAsyncMode(self) -> bool: ...
    @staticmethod
    def getCommonPoolParallelism() -> int: ...
    def getDelayedTaskCount(self) -> int: ...
    def getFactory(self) -> ForkJoinPool.ForkJoinWorkerThreadFactory: ...
    def getParallelism(self) -> int: ...
    def getPoolSize(self) -> int: ...
    def getQueuedSubmissionCount(self) -> int: ...
    def getQueuedTaskCount(self) -> int: ...
    def getRunningThreadCount(self) -> int: ...
    def getStealCount(self) -> int: ...
    def getUncaughtExceptionHandler(
        self,
    ) -> java.lang.Thread.UncaughtExceptionHandler: ...
    def hasQueuedSubmissions(self) -> bool: ...
    _invoke__T = typing.TypeVar("_invoke__T")  # <T>
    def invoke(self, forkJoinTask: ForkJoinTask[_invoke__T]) -> _invoke__T: ...
    _invokeAll_0__T = typing.TypeVar("_invokeAll_0__T")  # <T>
    _invokeAll_1__T = typing.TypeVar("_invokeAll_1__T")  # <T>
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_0__T]]
        | typing.Sequence[Callable[_invokeAll_0__T]]
        | set[Callable[_invokeAll_0__T]],
    ) -> java.util.List[Future[_invokeAll_0__T]]: ...
    @typing.overload
    def invokeAll(
        self,
        collection: java.util.Collection[Callable[_invokeAll_1__T]]
        | typing.Sequence[Callable[_invokeAll_1__T]]
        | set[Callable[_invokeAll_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> java.util.List[Future[_invokeAll_1__T]]: ...
    _invokeAllUninterruptibly__T = typing.TypeVar("_invokeAllUninterruptibly__T")  # <T>
    def invokeAllUninterruptibly(
        self,
        collection: java.util.Collection[Callable[_invokeAllUninterruptibly__T]]
        | typing.Sequence[Callable[_invokeAllUninterruptibly__T]]
        | set[Callable[_invokeAllUninterruptibly__T]],
    ) -> java.util.List[Future[_invokeAllUninterruptibly__T]]: ...
    _invokeAny_0__T = typing.TypeVar("_invokeAny_0__T")  # <T>
    _invokeAny_1__T = typing.TypeVar("_invokeAny_1__T")  # <T>
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_0__T]]
        | typing.Sequence[Callable[_invokeAny_0__T]]
        | set[Callable[_invokeAny_0__T]],
    ) -> _invokeAny_0__T: ...
    @typing.overload
    def invokeAny(
        self,
        collection: java.util.Collection[Callable[_invokeAny_1__T]]
        | typing.Sequence[Callable[_invokeAny_1__T]]
        | set[Callable[_invokeAny_1__T]],
        long: int,
        timeUnit: TimeUnit,
    ) -> _invokeAny_1__T: ...
    def isQuiescent(self) -> bool: ...
    def isShutdown(self) -> bool: ...
    def isTerminated(self) -> bool: ...
    def isTerminating(self) -> bool: ...
    _lazySubmit__T = typing.TypeVar("_lazySubmit__T")  # <T>
    def lazySubmit(
        self, forkJoinTask: ForkJoinTask[_lazySubmit__T]
    ) -> ForkJoinTask[_lazySubmit__T]: ...
    @staticmethod
    def managedBlock(managedBlocker: ForkJoinPool.ManagedBlocker) -> None: ...
    _schedule_1__V = typing.TypeVar("_schedule_1__V")  # <V>
    @typing.overload
    def schedule(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    @typing.overload
    def schedule(
        self,
        callable: Callable[_schedule_1__V] | typing.Callable[[], _schedule_1__V],
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[_schedule_1__V]: ...
    def scheduleAtFixedRate(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    def scheduleWithFixedDelay(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    def setParallelism(self, int: int) -> int: ...
    def shutdown(self) -> None: ...
    def shutdownNow(self) -> java.util.List[java.lang.Runnable]: ...
    _submit_1__T = typing.TypeVar("_submit_1__T")  # <T>
    _submit_2__T = typing.TypeVar("_submit_2__T")  # <T>
    _submit_3__T = typing.TypeVar("_submit_3__T")  # <T>
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> ForkJoinTask[typing.Any]: ...
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable, t: _submit_1__T
    ) -> ForkJoinTask[_submit_1__T]: ...
    @typing.overload
    def submit(
        self, callable: Callable[_submit_2__T] | typing.Callable[[], _submit_2__T]
    ) -> ForkJoinTask[_submit_2__T]: ...
    @typing.overload
    def submit(
        self, forkJoinTask: ForkJoinTask[_submit_3__T]
    ) -> ForkJoinTask[_submit_3__T]: ...
    _submitWithTimeout__V = typing.TypeVar("_submitWithTimeout__V")  # <V>
    def submitWithTimeout(
        self,
        callable: Callable[_submitWithTimeout__V]
        | typing.Callable[[], _submitWithTimeout__V],
        long: int,
        timeUnit: TimeUnit,
        consumer: java.util.function.Consumer[ForkJoinTask[_submitWithTimeout__V]]
        | typing.Callable[[ForkJoinTask[_submitWithTimeout__V]], None],
    ) -> ForkJoinTask[_submitWithTimeout__V]: ...
    def toString(self) -> java.lang.String: ...
    class ForkJoinWorkerThreadFactory:
        def newThread(self, forkJoinPool: ForkJoinPool) -> ForkJoinWorkerThread: ...

    class ManagedBlocker:
        def block(self) -> bool: ...
        def isReleasable(self) -> bool: ...

class ThreadPoolExecutor(AbstractExecutorService):
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        long: int,
        timeUnit: TimeUnit,
        blockingQueue: BlockingQueue[java.lang.Runnable | typing.Callable],
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        long: int,
        timeUnit: TimeUnit,
        blockingQueue: BlockingQueue[java.lang.Runnable | typing.Callable],
        rejectedExecutionHandler: RejectedExecutionHandler | typing.Callable,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        long: int,
        timeUnit: TimeUnit,
        blockingQueue: BlockingQueue[java.lang.Runnable | typing.Callable],
        threadFactory: ThreadFactory | typing.Callable,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        long: int,
        timeUnit: TimeUnit,
        blockingQueue: BlockingQueue[java.lang.Runnable | typing.Callable],
        threadFactory: ThreadFactory | typing.Callable,
        rejectedExecutionHandler: RejectedExecutionHandler | typing.Callable,
    ): ...
    def allowCoreThreadTimeOut(self, boolean: bool) -> None: ...
    def allowsCoreThreadTimeOut(self) -> bool: ...
    def awaitTermination(self, long: int, timeUnit: TimeUnit) -> bool: ...
    def execute(self, runnable: java.lang.Runnable | typing.Callable) -> None: ...
    def getActiveCount(self) -> int: ...
    def getCompletedTaskCount(self) -> int: ...
    def getCorePoolSize(self) -> int: ...
    def getKeepAliveTime(self, timeUnit: TimeUnit) -> int: ...
    def getLargestPoolSize(self) -> int: ...
    def getMaximumPoolSize(self) -> int: ...
    def getPoolSize(self) -> int: ...
    def getQueue(self) -> BlockingQueue[java.lang.Runnable]: ...
    def getRejectedExecutionHandler(self) -> RejectedExecutionHandler: ...
    def getTaskCount(self) -> int: ...
    def getThreadFactory(self) -> ThreadFactory: ...
    def isShutdown(self) -> bool: ...
    def isTerminated(self) -> bool: ...
    def isTerminating(self) -> bool: ...
    def prestartAllCoreThreads(self) -> int: ...
    def prestartCoreThread(self) -> bool: ...
    def purge(self) -> None: ...
    def remove(self, runnable: java.lang.Runnable | typing.Callable) -> bool: ...
    def setCorePoolSize(self, int: int) -> None: ...
    def setKeepAliveTime(self, long: int, timeUnit: TimeUnit) -> None: ...
    def setMaximumPoolSize(self, int: int) -> None: ...
    def setRejectedExecutionHandler(
        self, rejectedExecutionHandler: RejectedExecutionHandler | typing.Callable
    ) -> None: ...
    def setThreadFactory(
        self, threadFactory: ThreadFactory | typing.Callable
    ) -> None: ...
    def shutdown(self) -> None: ...
    def shutdownNow(self) -> java.util.List[java.lang.Runnable]: ...
    def toString(self) -> java.lang.String: ...
    class AbortPolicy(RejectedExecutionHandler):
        def __init__(self): ...
        def rejectedExecution(
            self,
            runnable: java.lang.Runnable | typing.Callable,
            threadPoolExecutor: ThreadPoolExecutor,
        ) -> None: ...

    class CallerRunsPolicy(RejectedExecutionHandler):
        def __init__(self): ...
        def rejectedExecution(
            self,
            runnable: java.lang.Runnable | typing.Callable,
            threadPoolExecutor: ThreadPoolExecutor,
        ) -> None: ...

    class DiscardOldestPolicy(RejectedExecutionHandler):
        def __init__(self): ...
        def rejectedExecution(
            self,
            runnable: java.lang.Runnable | typing.Callable,
            threadPoolExecutor: ThreadPoolExecutor,
        ) -> None: ...

    class DiscardPolicy(RejectedExecutionHandler):
        def __init__(self): ...
        def rejectedExecution(
            self,
            runnable: java.lang.Runnable | typing.Callable,
            threadPoolExecutor: ThreadPoolExecutor,
        ) -> None: ...

class ScheduledThreadPoolExecutor(ThreadPoolExecutor, ScheduledExecutorService):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        rejectedExecutionHandler: RejectedExecutionHandler | typing.Callable,
    ): ...
    @typing.overload
    def __init__(self, int: int, threadFactory: ThreadFactory | typing.Callable): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        threadFactory: ThreadFactory | typing.Callable,
        rejectedExecutionHandler: RejectedExecutionHandler | typing.Callable,
    ): ...
    def execute(self, runnable: java.lang.Runnable | typing.Callable) -> None: ...
    def getContinueExistingPeriodicTasksAfterShutdownPolicy(self) -> bool: ...
    def getExecuteExistingDelayedTasksAfterShutdownPolicy(self) -> bool: ...
    def getQueue(self) -> BlockingQueue[java.lang.Runnable]: ...
    def getRemoveOnCancelPolicy(self) -> bool: ...
    _schedule_1__V = typing.TypeVar("_schedule_1__V")  # <V>
    @typing.overload
    def schedule(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    @typing.overload
    def schedule(
        self,
        callable: Callable[_schedule_1__V] | typing.Callable[[], _schedule_1__V],
        long: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[_schedule_1__V]: ...
    def scheduleAtFixedRate(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    def scheduleWithFixedDelay(
        self,
        runnable: java.lang.Runnable | typing.Callable,
        long: int,
        long2: int,
        timeUnit: TimeUnit,
    ) -> ScheduledFuture[typing.Any]: ...
    def setContinueExistingPeriodicTasksAfterShutdownPolicy(
        self, boolean: bool
    ) -> None: ...
    def setExecuteExistingDelayedTasksAfterShutdownPolicy(
        self, boolean: bool
    ) -> None: ...
    def setRemoveOnCancelPolicy(self, boolean: bool) -> None: ...
    def shutdown(self) -> None: ...
    def shutdownNow(self) -> java.util.List[java.lang.Runnable]: ...
    _submit_1__T = typing.TypeVar("_submit_1__T")  # <T>
    _submit_2__T = typing.TypeVar("_submit_2__T")  # <T>
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> Future[typing.Any]: ...
    @typing.overload
    def submit(
        self, runnable: java.lang.Runnable | typing.Callable, t: _submit_1__T
    ) -> Future[_submit_1__T]: ...
    @typing.overload
    def submit(
        self, callable: Callable[_submit_2__T] | typing.Callable[[], _submit_2__T]
    ) -> Future[_submit_2__T]: ...

_ConcurrentHashMap__KeySetView__K = typing.TypeVar(
    "_ConcurrentHashMap__KeySetView__K"
)  # <K>
_ConcurrentHashMap__KeySetView__V = typing.TypeVar(
    "_ConcurrentHashMap__KeySetView__V"
)  # <V>
_ConcurrentHashMap__K = typing.TypeVar("_ConcurrentHashMap__K")  # <K>
_ConcurrentHashMap__V = typing.TypeVar("_ConcurrentHashMap__V")  # <V>

class ConcurrentHashMap(
    java.util.AbstractMap[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
    ConcurrentMap[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
    java.io.Serializable,
    typing.Generic[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, float: float): ...
    @typing.overload
    def __init__(self, int: int, float: float, int2: int): ...
    @typing.overload
    def __init__(
        self,
        map: java.util.Map[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        | typing.Mapping[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
    ): ...
    def clear(self) -> None: ...
    def compute(
        self,
        k: _ConcurrentHashMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _ConcurrentHashMap__V
        ]
        | typing.Callable[
            [_ConcurrentHashMap__K, _ConcurrentHashMap__V], _ConcurrentHashMap__V
        ],
    ) -> _ConcurrentHashMap__V: ...
    def computeIfAbsent(
        self,
        k: _ConcurrentHashMap__K,
        function: java.util.function.Function[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K], _ConcurrentHashMap__V],
    ) -> _ConcurrentHashMap__V: ...
    def computeIfPresent(
        self,
        k: _ConcurrentHashMap__K,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _ConcurrentHashMap__V
        ]
        | typing.Callable[
            [_ConcurrentHashMap__K, _ConcurrentHashMap__V], _ConcurrentHashMap__V
        ],
    ) -> _ConcurrentHashMap__V: ...
    def contains(self, object: typing.Any) -> bool: ...
    def containsKey(self, object: typing.Any) -> bool: ...
    def containsValue(self, object: typing.Any) -> bool: ...
    def elements(self) -> java.util.Enumeration[_ConcurrentHashMap__V]: ...
    def entrySet(
        self,
    ) -> java.util.Set[
        java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
    ]: ...
    def equals(self, object: typing.Any) -> bool: ...
    _forEach_2__U = typing.TypeVar("_forEach_2__U")  # <U>
    @typing.overload
    def forEach(
        self,
        biConsumer: java.util.function.BiConsumer[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], None],
    ) -> None: ...
    @typing.overload
    def forEach(
        self,
        long: int,
        biConsumer: java.util.function.BiConsumer[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], None],
    ) -> None: ...
    @typing.overload
    def forEach(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _forEach_2__U
        ]
        | typing.Callable[
            [_ConcurrentHashMap__K, _ConcurrentHashMap__V], _forEach_2__U
        ],
        consumer: java.util.function.Consumer[_forEach_2__U]
        | typing.Callable[[_forEach_2__U], None],
    ) -> None: ...
    _forEachEntry_1__U = typing.TypeVar("_forEachEntry_1__U")  # <U>
    @typing.overload
    def forEachEntry(
        self,
        long: int,
        consumer: java.util.function.Consumer[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]], None
        ],
    ) -> None: ...
    @typing.overload
    def forEachEntry(
        self,
        long: int,
        function: java.util.function.Function[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            _forEachEntry_1__U,
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]],
            _forEachEntry_1__U,
        ],
        consumer: java.util.function.Consumer[_forEachEntry_1__U]
        | typing.Callable[[_forEachEntry_1__U], None],
    ) -> None: ...
    _forEachKey_1__U = typing.TypeVar("_forEachKey_1__U")  # <U>
    @typing.overload
    def forEachKey(
        self,
        long: int,
        consumer: java.util.function.Consumer[_ConcurrentHashMap__K]
        | typing.Callable[[_ConcurrentHashMap__K], None],
    ) -> None: ...
    @typing.overload
    def forEachKey(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__K, _forEachKey_1__U]
        | typing.Callable[[_ConcurrentHashMap__K], _forEachKey_1__U],
        consumer: java.util.function.Consumer[_forEachKey_1__U]
        | typing.Callable[[_forEachKey_1__U], None],
    ) -> None: ...
    _forEachValue_1__U = typing.TypeVar("_forEachValue_1__U")  # <U>
    @typing.overload
    def forEachValue(
        self,
        long: int,
        consumer: java.util.function.Consumer[_ConcurrentHashMap__V]
        | typing.Callable[[_ConcurrentHashMap__V], None],
    ) -> None: ...
    @typing.overload
    def forEachValue(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__V, _forEachValue_1__U]
        | typing.Callable[[_ConcurrentHashMap__V], _forEachValue_1__U],
        consumer: java.util.function.Consumer[_forEachValue_1__U]
        | typing.Callable[[_forEachValue_1__U], None],
    ) -> None: ...
    def get(self, object: typing.Any) -> _ConcurrentHashMap__V: ...
    def getOrDefault(
        self, object: typing.Any, v: _ConcurrentHashMap__V
    ) -> _ConcurrentHashMap__V: ...
    def hashCode(self) -> int: ...
    def isEmpty(self) -> bool: ...
    @typing.overload
    def keySet(
        self,
    ) -> ConcurrentHashMap.KeySetView[_ConcurrentHashMap__K, _ConcurrentHashMap__V]: ...
    @typing.overload
    def keySet(
        self, v: _ConcurrentHashMap__V
    ) -> ConcurrentHashMap.KeySetView[_ConcurrentHashMap__K, _ConcurrentHashMap__V]: ...
    def keys(self) -> java.util.Enumeration[_ConcurrentHashMap__K]: ...
    def mappingCount(self) -> int: ...
    def merge(
        self,
        k: _ConcurrentHashMap__K,
        v: _ConcurrentHashMap__V,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__V, _ConcurrentHashMap__V, _ConcurrentHashMap__V
        ]
        | typing.Callable[
            [_ConcurrentHashMap__V, _ConcurrentHashMap__V], _ConcurrentHashMap__V
        ],
    ) -> _ConcurrentHashMap__V: ...
    _newKeySet_0__K = typing.TypeVar("_newKeySet_0__K")  # <K>
    _newKeySet_1__K = typing.TypeVar("_newKeySet_1__K")  # <K>
    @typing.overload
    @staticmethod
    def newKeySet() -> ConcurrentHashMap.KeySetView[_newKeySet_0__K, bool]: ...
    @typing.overload
    @staticmethod
    def newKeySet(int: int) -> ConcurrentHashMap.KeySetView[_newKeySet_1__K, bool]: ...
    def put(
        self, k: _ConcurrentHashMap__K, v: _ConcurrentHashMap__V
    ) -> _ConcurrentHashMap__V: ...
    def putAll(
        self,
        map: java.util.Map[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        | typing.Mapping[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
    ) -> None: ...
    def putIfAbsent(
        self, k: _ConcurrentHashMap__K, v: _ConcurrentHashMap__V
    ) -> _ConcurrentHashMap__V: ...
    _reduce__U = typing.TypeVar("_reduce__U")  # <U>
    def reduce(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _reduce__U
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], _reduce__U],
        biFunction2: java.util.function.BiFunction[_reduce__U, _reduce__U, _reduce__U]
        | typing.Callable[[_reduce__U, _reduce__U], _reduce__U],
    ) -> _reduce__U: ...
    _reduceEntries_0__U = typing.TypeVar("_reduceEntries_0__U")  # <U>
    @typing.overload
    def reduceEntries(
        self,
        long: int,
        function: java.util.function.Function[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            _reduceEntries_0__U,
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]],
            _reduceEntries_0__U,
        ],
        biFunction: java.util.function.BiFunction[
            _reduceEntries_0__U, _reduceEntries_0__U, _reduceEntries_0__U
        ]
        | typing.Callable[
            [_reduceEntries_0__U, _reduceEntries_0__U], _reduceEntries_0__U
        ],
    ) -> _reduceEntries_0__U: ...
    @typing.overload
    def reduceEntries(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
        ]
        | typing.Callable[
            [
                java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
                java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            ],
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
        ],
    ) -> java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]: ...
    def reduceEntriesToDouble(
        self,
        long: int,
        toDoubleFunction: java.util.function.ToDoubleFunction[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]], float
        ],
        double: float,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> float: ...
    def reduceEntriesToInt(
        self,
        long: int,
        toIntFunction: java.util.function.ToIntFunction[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]], int
        ],
        int: int,
        intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable,
    ) -> int: ...
    def reduceEntriesToLong(
        self,
        long: int,
        toLongFunction: java.util.function.ToLongFunction[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]], int
        ],
        long2: int,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> int: ...
    _reduceKeys_1__U = typing.TypeVar("_reduceKeys_1__U")  # <U>
    @typing.overload
    def reduceKeys(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__K, _ConcurrentHashMap__K
        ]
        | typing.Callable[
            [_ConcurrentHashMap__K, _ConcurrentHashMap__K], _ConcurrentHashMap__K
        ],
    ) -> _ConcurrentHashMap__K: ...
    @typing.overload
    def reduceKeys(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__K, _reduceKeys_1__U]
        | typing.Callable[[_ConcurrentHashMap__K], _reduceKeys_1__U],
        biFunction: java.util.function.BiFunction[
            _reduceKeys_1__U, _reduceKeys_1__U, _reduceKeys_1__U
        ]
        | typing.Callable[[_reduceKeys_1__U, _reduceKeys_1__U], _reduceKeys_1__U],
    ) -> _reduceKeys_1__U: ...
    def reduceKeysToDouble(
        self,
        long: int,
        toDoubleFunction: java.util.function.ToDoubleFunction[_ConcurrentHashMap__K]
        | typing.Callable[[_ConcurrentHashMap__K], float],
        double: float,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> float: ...
    def reduceKeysToInt(
        self,
        long: int,
        toIntFunction: java.util.function.ToIntFunction[_ConcurrentHashMap__K]
        | typing.Callable[[_ConcurrentHashMap__K], int],
        int: int,
        intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable,
    ) -> int: ...
    def reduceKeysToLong(
        self,
        long: int,
        toLongFunction: java.util.function.ToLongFunction[_ConcurrentHashMap__K]
        | typing.Callable[[_ConcurrentHashMap__K], int],
        long2: int,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> int: ...
    def reduceToDouble(
        self,
        long: int,
        toDoubleBiFunction: java.util.function.ToDoubleBiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], float],
        double: float,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> float: ...
    def reduceToInt(
        self,
        long: int,
        toIntBiFunction: java.util.function.ToIntBiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], int],
        int: int,
        intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable,
    ) -> int: ...
    def reduceToLong(
        self,
        long: int,
        toLongBiFunction: java.util.function.ToLongBiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], int],
        long2: int,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> int: ...
    _reduceValues_1__U = typing.TypeVar("_reduceValues_1__U")  # <U>
    @typing.overload
    def reduceValues(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__V, _ConcurrentHashMap__V, _ConcurrentHashMap__V
        ]
        | typing.Callable[
            [_ConcurrentHashMap__V, _ConcurrentHashMap__V], _ConcurrentHashMap__V
        ],
    ) -> _ConcurrentHashMap__V: ...
    @typing.overload
    def reduceValues(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__V, _reduceValues_1__U]
        | typing.Callable[[_ConcurrentHashMap__V], _reduceValues_1__U],
        biFunction: java.util.function.BiFunction[
            _reduceValues_1__U, _reduceValues_1__U, _reduceValues_1__U
        ]
        | typing.Callable[[_reduceValues_1__U, _reduceValues_1__U], _reduceValues_1__U],
    ) -> _reduceValues_1__U: ...
    def reduceValuesToDouble(
        self,
        long: int,
        toDoubleFunction: java.util.function.ToDoubleFunction[_ConcurrentHashMap__V]
        | typing.Callable[[_ConcurrentHashMap__V], float],
        double: float,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> float: ...
    def reduceValuesToInt(
        self,
        long: int,
        toIntFunction: java.util.function.ToIntFunction[_ConcurrentHashMap__V]
        | typing.Callable[[_ConcurrentHashMap__V], int],
        int: int,
        intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable,
    ) -> int: ...
    def reduceValuesToLong(
        self,
        long: int,
        toLongFunction: java.util.function.ToLongFunction[_ConcurrentHashMap__V]
        | typing.Callable[[_ConcurrentHashMap__V], int],
        long2: int,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> int: ...
    @typing.overload
    def remove(self, object: typing.Any, object2: typing.Any) -> bool: ...
    @typing.overload
    def remove(self, object: typing.Any) -> _ConcurrentHashMap__V: ...
    @typing.overload
    def replace(
        self,
        k: _ConcurrentHashMap__K,
        v: _ConcurrentHashMap__V,
        v2: _ConcurrentHashMap__V,
    ) -> bool: ...
    @typing.overload
    def replace(
        self, k: _ConcurrentHashMap__K, v: _ConcurrentHashMap__V
    ) -> _ConcurrentHashMap__V: ...
    def replaceAll(
        self,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _ConcurrentHashMap__V
        ]
        | typing.Callable[
            [_ConcurrentHashMap__K, _ConcurrentHashMap__V], _ConcurrentHashMap__V
        ],
    ) -> None: ...
    _search__U = typing.TypeVar("_search__U")  # <U>
    def search(
        self,
        long: int,
        biFunction: java.util.function.BiFunction[
            _ConcurrentHashMap__K, _ConcurrentHashMap__V, _search__U
        ]
        | typing.Callable[[_ConcurrentHashMap__K, _ConcurrentHashMap__V], _search__U],
    ) -> _search__U: ...
    _searchEntries__U = typing.TypeVar("_searchEntries__U")  # <U>
    def searchEntries(
        self,
        long: int,
        function: java.util.function.Function[
            java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V],
            _searchEntries__U,
        ]
        | typing.Callable[
            [java.util.Map.Entry[_ConcurrentHashMap__K, _ConcurrentHashMap__V]],
            _searchEntries__U,
        ],
    ) -> _searchEntries__U: ...
    _searchKeys__U = typing.TypeVar("_searchKeys__U")  # <U>
    def searchKeys(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__K, _searchKeys__U]
        | typing.Callable[[_ConcurrentHashMap__K], _searchKeys__U],
    ) -> _searchKeys__U: ...
    _searchValues__U = typing.TypeVar("_searchValues__U")  # <U>
    def searchValues(
        self,
        long: int,
        function: java.util.function.Function[_ConcurrentHashMap__V, _searchValues__U]
        | typing.Callable[[_ConcurrentHashMap__V], _searchValues__U],
    ) -> _searchValues__U: ...
    def size(self) -> int: ...
    def toString(self) -> java.lang.String: ...
    def values(self) -> java.util.Collection[_ConcurrentHashMap__V]: ...
    class KeySetView(
        java.util.concurrent.ConcurrentHashMap.CollectionView[
            _ConcurrentHashMap__KeySetView__K,
            _ConcurrentHashMap__KeySetView__V,
            _ConcurrentHashMap__KeySetView__K,
        ],
        java.util.Set[_ConcurrentHashMap__KeySetView__K],
        java.io.Serializable,
        typing.Generic[
            _ConcurrentHashMap__KeySetView__K, _ConcurrentHashMap__KeySetView__V
        ],
    ):
        def add(self, k: _ConcurrentHashMap__KeySetView__K) -> bool: ...
        def addAll(
            self,
            collection: java.util.Collection[_ConcurrentHashMap__KeySetView__K]
            | typing.Sequence[_ConcurrentHashMap__KeySetView__K]
            | set[_ConcurrentHashMap__KeySetView__K],
        ) -> bool: ...
        def contains(self, object: typing.Any) -> bool: ...
        def equals(self, object: typing.Any) -> bool: ...
        def forEach(
            self,
            consumer: java.util.function.Consumer[_ConcurrentHashMap__KeySetView__K]
            | typing.Callable[[_ConcurrentHashMap__KeySetView__K], None],
        ) -> None: ...
        def getMappedValue(self) -> _ConcurrentHashMap__KeySetView__V: ...
        def hashCode(self) -> int: ...
        def iterator(self) -> java.util.Iterator[_ConcurrentHashMap__KeySetView__K]: ...
        def remove(self, object: typing.Any) -> bool: ...
        def spliterator(
            self,
        ) -> java.util.Spliterator[_ConcurrentHashMap__KeySetView__K]: ...

    class CollectionView: ...

_Flow__Processor__T = typing.TypeVar("_Flow__Processor__T")  # <T>
_Flow__Processor__R = typing.TypeVar("_Flow__Processor__R")  # <R>
_Flow__Publisher__T = typing.TypeVar("_Flow__Publisher__T")  # <T>
_Flow__Subscriber__T = typing.TypeVar("_Flow__Subscriber__T")  # <T>

class Flow:
    @staticmethod
    def defaultBufferSize() -> int: ...
    class Processor(
        java.util.concurrent.Flow.Subscriber[_Flow__Processor__T],
        java.util.concurrent.Flow.Publisher[_Flow__Processor__R],
        typing.Generic[_Flow__Processor__T, _Flow__Processor__R],
    ): ...

    class Publisher(typing.Generic[_Flow__Publisher__T]):
        def subscribe(
            self, subscriber: Flow.Subscriber[_Flow__Publisher__T]
        ) -> None: ...

    class Subscriber(typing.Generic[_Flow__Subscriber__T]):
        def onComplete(self) -> None: ...
        def onError(self, throwable: java.lang.Throwable) -> None: ...
        def onNext(self, t: _Flow__Subscriber__T) -> None: ...
        def onSubscribe(self, subscription: Flow.Subscription) -> None: ...

    class Subscription:
        def cancel(self) -> None: ...
        def request(self, long: int) -> None: ...

_SubmissionPublisher__T = typing.TypeVar("_SubmissionPublisher__T")  # <T>

class SubmissionPublisher(
    Flow.Publisher[_SubmissionPublisher__T],
    java.lang.AutoCloseable,
    typing.Generic[_SubmissionPublisher__T],
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, executor: Executor | typing.Callable, int: int): ...
    @typing.overload
    def __init__(
        self,
        executor: Executor | typing.Callable,
        int: int,
        biConsumer: java.util.function.BiConsumer[
            Flow.Subscriber[_SubmissionPublisher__T], java.lang.Throwable
        ]
        | typing.Callable[
            [Flow.Subscriber[_SubmissionPublisher__T], java.lang.Throwable], None
        ],
    ): ...
    def close(self) -> None: ...
    def closeExceptionally(self, throwable: java.lang.Throwable) -> None: ...
    def consume(
        self,
        consumer: java.util.function.Consumer[_SubmissionPublisher__T]
        | typing.Callable[[_SubmissionPublisher__T], None],
    ) -> CompletableFuture[None]: ...
    def estimateMaximumLag(self) -> int: ...
    def estimateMinimumDemand(self) -> int: ...
    def getClosedException(self) -> java.lang.Throwable: ...
    def getExecutor(self) -> Executor: ...
    def getMaxBufferCapacity(self) -> int: ...
    def getNumberOfSubscribers(self) -> int: ...
    def getSubscribers(
        self,
    ) -> java.util.List[Flow.Subscriber[_SubmissionPublisher__T]]: ...
    def hasSubscribers(self) -> bool: ...
    def isClosed(self) -> bool: ...
    def isSubscribed(
        self, subscriber: Flow.Subscriber[_SubmissionPublisher__T]
    ) -> bool: ...
    @typing.overload
    def offer(
        self,
        t: _SubmissionPublisher__T,
        biPredicate: java.util.function.BiPredicate[
            Flow.Subscriber[_SubmissionPublisher__T], _SubmissionPublisher__T
        ]
        | typing.Callable[
            [Flow.Subscriber[_SubmissionPublisher__T], _SubmissionPublisher__T], bool
        ],
    ) -> int: ...
    @typing.overload
    def offer(
        self,
        t: _SubmissionPublisher__T,
        long: int,
        timeUnit: TimeUnit,
        biPredicate: java.util.function.BiPredicate[
            Flow.Subscriber[_SubmissionPublisher__T], _SubmissionPublisher__T
        ]
        | typing.Callable[
            [Flow.Subscriber[_SubmissionPublisher__T], _SubmissionPublisher__T], bool
        ],
    ) -> int: ...
    def submit(self, t: _SubmissionPublisher__T) -> int: ...
    def subscribe(
        self, subscriber: Flow.Subscriber[_SubmissionPublisher__T]
    ) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.util.concurrent")``.

    AbstractExecutorService: type[AbstractExecutorService]
    ArrayBlockingQueue: type[ArrayBlockingQueue]
    BlockingDeque: type[BlockingDeque]
    BlockingQueue: type[BlockingQueue]
    BrokenBarrierException: type[BrokenBarrierException]
    Callable: type[Callable]
    CancellationException: type[CancellationException]
    CompletableFuture: type[CompletableFuture]
    CompletionException: type[CompletionException]
    CompletionService: type[CompletionService]
    CompletionStage: type[CompletionStage]
    ConcurrentHashMap: type[ConcurrentHashMap]
    ConcurrentLinkedDeque: type[ConcurrentLinkedDeque]
    ConcurrentLinkedQueue: type[ConcurrentLinkedQueue]
    ConcurrentMap: type[ConcurrentMap]
    ConcurrentNavigableMap: type[ConcurrentNavigableMap]
    ConcurrentSkipListMap: type[ConcurrentSkipListMap]
    ConcurrentSkipListSet: type[ConcurrentSkipListSet]
    CopyOnWriteArrayList: type[CopyOnWriteArrayList]
    CopyOnWriteArraySet: type[CopyOnWriteArraySet]
    CountDownLatch: type[CountDownLatch]
    CountedCompleter: type[CountedCompleter]
    CyclicBarrier: type[CyclicBarrier]
    DelayQueue: type[DelayQueue]
    Delayed: type[Delayed]
    Exchanger: type[Exchanger]
    ExecutionException: type[ExecutionException]
    Executor: type[Executor]
    ExecutorCompletionService: type[ExecutorCompletionService]
    ExecutorService: type[ExecutorService]
    Executors: type[Executors]
    Flow: type[Flow]
    ForkJoinPool: type[ForkJoinPool]
    ForkJoinTask: type[ForkJoinTask]
    ForkJoinWorkerThread: type[ForkJoinWorkerThread]
    Future: type[Future]
    FutureTask: type[FutureTask]
    LinkedBlockingDeque: type[LinkedBlockingDeque]
    LinkedBlockingQueue: type[LinkedBlockingQueue]
    LinkedTransferQueue: type[LinkedTransferQueue]
    Phaser: type[Phaser]
    PriorityBlockingQueue: type[PriorityBlockingQueue]
    RecursiveAction: type[RecursiveAction]
    RecursiveTask: type[RecursiveTask]
    RejectedExecutionException: type[RejectedExecutionException]
    RejectedExecutionHandler: type[RejectedExecutionHandler]
    RunnableFuture: type[RunnableFuture]
    RunnableScheduledFuture: type[RunnableScheduledFuture]
    ScheduledExecutorService: type[ScheduledExecutorService]
    ScheduledFuture: type[ScheduledFuture]
    ScheduledThreadPoolExecutor: type[ScheduledThreadPoolExecutor]
    Semaphore: type[Semaphore]
    StructureViolationException: type[StructureViolationException]
    StructuredTaskScope: type[StructuredTaskScope]
    SubmissionPublisher: type[SubmissionPublisher]
    SynchronousQueue: type[SynchronousQueue]
    ThreadFactory: type[ThreadFactory]
    ThreadLocalRandom: type[ThreadLocalRandom]
    ThreadPoolExecutor: type[ThreadPoolExecutor]
    TimeUnit: type[TimeUnit]
    TimeoutException: type[TimeoutException]
    TransferQueue: type[TransferQueue]
    atomic: java.util.concurrent.atomic.__module_protocol__
    locks: java.util.concurrent.locks.__module_protocol__
