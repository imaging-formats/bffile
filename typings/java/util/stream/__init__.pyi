import typing
from typing import Protocol

import java
import java.lang
import java.util
import java.util.concurrent
import java.util.function
import jpype

_BaseStream__T = typing.TypeVar("_BaseStream__T")  # <T>
_BaseStream__S = typing.TypeVar("_BaseStream__S", bound="BaseStream")  # <S>

class BaseStream(
    java.lang.AutoCloseable, typing.Generic[_BaseStream__T, _BaseStream__S]
):
    def close(self) -> None: ...
    def isParallel(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[_BaseStream__T]: ...
    def onClose(
        self, runnable: java.lang.Runnable | typing.Callable
    ) -> _BaseStream__S: ...
    def parallel(self) -> _BaseStream__S: ...
    def sequential(self) -> _BaseStream__S: ...
    def spliterator(self) -> java.util.Spliterator[_BaseStream__T]: ...
    def unordered(self) -> _BaseStream__S: ...

_Collector__T = typing.TypeVar("_Collector__T")  # <T>
_Collector__A = typing.TypeVar("_Collector__A")  # <A>
_Collector__R = typing.TypeVar("_Collector__R")  # <R>

class Collector(typing.Generic[_Collector__T, _Collector__A, _Collector__R]):
    def accumulator(
        self,
    ) -> java.util.function.BiConsumer[_Collector__A, _Collector__T]: ...
    def characteristics(self) -> java.util.Set[Collector.Characteristics]: ...
    def combiner(self) -> java.util.function.BinaryOperator[_Collector__A]: ...
    def finisher(self) -> java.util.function.Function[_Collector__A, _Collector__R]: ...
    _of_0__T = typing.TypeVar("_of_0__T")  # <T>
    _of_0__A = typing.TypeVar("_of_0__A")  # <A>
    _of_0__R = typing.TypeVar("_of_0__R")  # <R>
    _of_1__T = typing.TypeVar("_of_1__T")  # <T>
    _of_1__R = typing.TypeVar("_of_1__R")  # <R>
    @typing.overload
    @staticmethod
    def of(
        supplier: java.util.function.Supplier[_of_0__A] | typing.Callable[[], _of_0__A],
        biConsumer: java.util.function.BiConsumer[_of_0__A, _of_0__T]
        | typing.Callable[[_of_0__A, _of_0__T], None],
        binaryOperator: java.util.function.BinaryOperator[_of_0__A] | typing.Callable,
        function: java.util.function.Function[_of_0__A, _of_0__R]
        | typing.Callable[[_of_0__A], _of_0__R],
        *characteristics: Collector.Characteristics,
    ) -> Collector[_of_0__T, _of_0__A, _of_0__R]: ...
    @typing.overload
    @staticmethod
    def of(
        supplier: java.util.function.Supplier[_of_1__R] | typing.Callable[[], _of_1__R],
        biConsumer: java.util.function.BiConsumer[_of_1__R, _of_1__T]
        | typing.Callable[[_of_1__R, _of_1__T], None],
        binaryOperator: java.util.function.BinaryOperator[_of_1__R] | typing.Callable,
        *characteristics: Collector.Characteristics,
    ) -> Collector[_of_1__T, _of_1__R, _of_1__R]: ...
    def supplier(self) -> java.util.function.Supplier[_Collector__A]: ...
    class Characteristics(java.lang.Enum["Collector.Characteristics"]):
        CONCURRENT: typing.ClassVar[Collector.Characteristics] = ...
        UNORDERED: typing.ClassVar[Collector.Characteristics] = ...
        IDENTITY_FINISH: typing.ClassVar[Collector.Characteristics] = ...
        _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_0__T], string: java.lang.String | str
        ) -> _valueOf_0__T: ...
        @typing.overload
        @staticmethod
        def valueOf(string: java.lang.String | str) -> Collector.Characteristics: ...
        @staticmethod
        def values() -> typing.MutableSequence[Collector.Characteristics]: ...

class Collectors:
    _averagingDouble__T = typing.TypeVar("_averagingDouble__T")  # <T>
    @staticmethod
    def averagingDouble(
        toDoubleFunction: java.util.function.ToDoubleFunction[_averagingDouble__T]
        | typing.Callable[[_averagingDouble__T], float],
    ) -> Collector[_averagingDouble__T, typing.Any, float]: ...
    _averagingInt__T = typing.TypeVar("_averagingInt__T")  # <T>
    @staticmethod
    def averagingInt(
        toIntFunction: java.util.function.ToIntFunction[_averagingInt__T]
        | typing.Callable[[_averagingInt__T], int],
    ) -> Collector[_averagingInt__T, typing.Any, float]: ...
    _averagingLong__T = typing.TypeVar("_averagingLong__T")  # <T>
    @staticmethod
    def averagingLong(
        toLongFunction: java.util.function.ToLongFunction[_averagingLong__T]
        | typing.Callable[[_averagingLong__T], int],
    ) -> Collector[_averagingLong__T, typing.Any, float]: ...
    _collectingAndThen__T = typing.TypeVar("_collectingAndThen__T")  # <T>
    _collectingAndThen__A = typing.TypeVar("_collectingAndThen__A")  # <A>
    _collectingAndThen__R = typing.TypeVar("_collectingAndThen__R")  # <R>
    _collectingAndThen__RR = typing.TypeVar("_collectingAndThen__RR")  # <RR>
    @staticmethod
    def collectingAndThen(
        collector: Collector[
            _collectingAndThen__T, _collectingAndThen__A, _collectingAndThen__R
        ],
        function: java.util.function.Function[
            _collectingAndThen__R, _collectingAndThen__RR
        ]
        | typing.Callable[[_collectingAndThen__R], _collectingAndThen__RR],
    ) -> Collector[
        _collectingAndThen__T, _collectingAndThen__A, _collectingAndThen__RR
    ]: ...
    _counting__T = typing.TypeVar("_counting__T")  # <T>
    @staticmethod
    def counting() -> Collector[_counting__T, typing.Any, int]: ...
    _filtering__T = typing.TypeVar("_filtering__T")  # <T>
    _filtering__A = typing.TypeVar("_filtering__A")  # <A>
    _filtering__R = typing.TypeVar("_filtering__R")  # <R>
    @staticmethod
    def filtering(
        predicate: java.util.function.Predicate[_filtering__T]
        | typing.Callable[[_filtering__T], bool],
        collector: Collector[_filtering__T, _filtering__A, _filtering__R],
    ) -> Collector[_filtering__T, typing.Any, _filtering__R]: ...
    _flatMapping__T = typing.TypeVar("_flatMapping__T")  # <T>
    _flatMapping__U = typing.TypeVar("_flatMapping__U")  # <U>
    _flatMapping__A = typing.TypeVar("_flatMapping__A")  # <A>
    _flatMapping__R = typing.TypeVar("_flatMapping__R")  # <R>
    @staticmethod
    def flatMapping(
        function: java.util.function.Function[_flatMapping__T, Stream[_flatMapping__U]]
        | typing.Callable[[_flatMapping__T], Stream[_flatMapping__U]],
        collector: Collector[_flatMapping__U, _flatMapping__A, _flatMapping__R],
    ) -> Collector[_flatMapping__T, typing.Any, _flatMapping__R]: ...
    _groupingBy_0__T = typing.TypeVar("_groupingBy_0__T")  # <T>
    _groupingBy_0__K = typing.TypeVar("_groupingBy_0__K")  # <K>
    _groupingBy_1__T = typing.TypeVar("_groupingBy_1__T")  # <T>
    _groupingBy_1__K = typing.TypeVar("_groupingBy_1__K")  # <K>
    _groupingBy_1__D = typing.TypeVar("_groupingBy_1__D")  # <D>
    _groupingBy_1__A = typing.TypeVar("_groupingBy_1__A")  # <A>
    _groupingBy_1__M = typing.TypeVar("_groupingBy_1__M", bound=java.util.Map)  # <M>
    _groupingBy_2__T = typing.TypeVar("_groupingBy_2__T")  # <T>
    _groupingBy_2__K = typing.TypeVar("_groupingBy_2__K")  # <K>
    _groupingBy_2__A = typing.TypeVar("_groupingBy_2__A")  # <A>
    _groupingBy_2__D = typing.TypeVar("_groupingBy_2__D")  # <D>
    @typing.overload
    @staticmethod
    def groupingBy(
        function: java.util.function.Function[_groupingBy_0__T, _groupingBy_0__K]
        | typing.Callable[[_groupingBy_0__T], _groupingBy_0__K],
    ) -> Collector[
        _groupingBy_0__T,
        typing.Any,
        java.util.Map[_groupingBy_0__K, java.util.List[_groupingBy_0__T]],
    ]: ...
    @typing.overload
    @staticmethod
    def groupingBy(
        function: java.util.function.Function[_groupingBy_1__T, _groupingBy_1__K]
        | typing.Callable[[_groupingBy_1__T], _groupingBy_1__K],
        supplier: java.util.function.Supplier[_groupingBy_1__M]
        | typing.Callable[[], _groupingBy_1__M],
        collector: Collector[_groupingBy_1__T, _groupingBy_1__A, _groupingBy_1__D],
    ) -> Collector[_groupingBy_1__T, typing.Any, _groupingBy_1__M]: ...
    @typing.overload
    @staticmethod
    def groupingBy(
        function: java.util.function.Function[_groupingBy_2__T, _groupingBy_2__K]
        | typing.Callable[[_groupingBy_2__T], _groupingBy_2__K],
        collector: Collector[_groupingBy_2__T, _groupingBy_2__A, _groupingBy_2__D],
    ) -> Collector[
        _groupingBy_2__T, typing.Any, java.util.Map[_groupingBy_2__K, _groupingBy_2__D]
    ]: ...
    _groupingByConcurrent_0__T = typing.TypeVar("_groupingByConcurrent_0__T")  # <T>
    _groupingByConcurrent_0__K = typing.TypeVar("_groupingByConcurrent_0__K")  # <K>
    _groupingByConcurrent_1__T = typing.TypeVar("_groupingByConcurrent_1__T")  # <T>
    _groupingByConcurrent_1__K = typing.TypeVar("_groupingByConcurrent_1__K")  # <K>
    _groupingByConcurrent_1__A = typing.TypeVar("_groupingByConcurrent_1__A")  # <A>
    _groupingByConcurrent_1__D = typing.TypeVar("_groupingByConcurrent_1__D")  # <D>
    _groupingByConcurrent_1__M = typing.TypeVar(
        "_groupingByConcurrent_1__M", bound=java.util.concurrent.ConcurrentMap
    )  # <M>
    _groupingByConcurrent_2__T = typing.TypeVar("_groupingByConcurrent_2__T")  # <T>
    _groupingByConcurrent_2__K = typing.TypeVar("_groupingByConcurrent_2__K")  # <K>
    _groupingByConcurrent_2__A = typing.TypeVar("_groupingByConcurrent_2__A")  # <A>
    _groupingByConcurrent_2__D = typing.TypeVar("_groupingByConcurrent_2__D")  # <D>
    @typing.overload
    @staticmethod
    def groupingByConcurrent(
        function: java.util.function.Function[
            _groupingByConcurrent_0__T, _groupingByConcurrent_0__K
        ]
        | typing.Callable[[_groupingByConcurrent_0__T], _groupingByConcurrent_0__K],
    ) -> Collector[
        _groupingByConcurrent_0__T,
        typing.Any,
        java.util.concurrent.ConcurrentMap[
            _groupingByConcurrent_0__K, java.util.List[_groupingByConcurrent_0__T]
        ],
    ]: ...
    @typing.overload
    @staticmethod
    def groupingByConcurrent(
        function: java.util.function.Function[
            _groupingByConcurrent_1__T, _groupingByConcurrent_1__K
        ]
        | typing.Callable[[_groupingByConcurrent_1__T], _groupingByConcurrent_1__K],
        supplier: java.util.function.Supplier[_groupingByConcurrent_1__M]
        | typing.Callable[[], _groupingByConcurrent_1__M],
        collector: Collector[
            _groupingByConcurrent_1__T,
            _groupingByConcurrent_1__A,
            _groupingByConcurrent_1__D,
        ],
    ) -> Collector[
        _groupingByConcurrent_1__T, typing.Any, _groupingByConcurrent_1__M
    ]: ...
    @typing.overload
    @staticmethod
    def groupingByConcurrent(
        function: java.util.function.Function[
            _groupingByConcurrent_2__T, _groupingByConcurrent_2__K
        ]
        | typing.Callable[[_groupingByConcurrent_2__T], _groupingByConcurrent_2__K],
        collector: Collector[
            _groupingByConcurrent_2__T,
            _groupingByConcurrent_2__A,
            _groupingByConcurrent_2__D,
        ],
    ) -> Collector[
        _groupingByConcurrent_2__T,
        typing.Any,
        java.util.concurrent.ConcurrentMap[
            _groupingByConcurrent_2__K, _groupingByConcurrent_2__D
        ],
    ]: ...
    @typing.overload
    @staticmethod
    def joining() -> Collector[
        java.lang.CharSequence, typing.Any, java.lang.String
    ]: ...
    @typing.overload
    @staticmethod
    def joining(
        charSequence: java.lang.CharSequence | str,
    ) -> Collector[java.lang.CharSequence, typing.Any, java.lang.String]: ...
    @typing.overload
    @staticmethod
    def joining(
        charSequence: java.lang.CharSequence | str,
        charSequence2: java.lang.CharSequence | str,
        charSequence3: java.lang.CharSequence | str,
    ) -> Collector[java.lang.CharSequence, typing.Any, java.lang.String]: ...
    _mapping__T = typing.TypeVar("_mapping__T")  # <T>
    _mapping__U = typing.TypeVar("_mapping__U")  # <U>
    _mapping__A = typing.TypeVar("_mapping__A")  # <A>
    _mapping__R = typing.TypeVar("_mapping__R")  # <R>
    @staticmethod
    def mapping(
        function: java.util.function.Function[_mapping__T, _mapping__U]
        | typing.Callable[[_mapping__T], _mapping__U],
        collector: Collector[_mapping__U, _mapping__A, _mapping__R],
    ) -> Collector[_mapping__T, typing.Any, _mapping__R]: ...
    _maxBy__T = typing.TypeVar("_maxBy__T")  # <T>
    @staticmethod
    def maxBy(
        comparator: java.util.Comparator[_maxBy__T]
        | typing.Callable[[_maxBy__T, _maxBy__T], int],
    ) -> Collector[_maxBy__T, typing.Any, java.util.Optional[_maxBy__T]]: ...
    _minBy__T = typing.TypeVar("_minBy__T")  # <T>
    @staticmethod
    def minBy(
        comparator: java.util.Comparator[_minBy__T]
        | typing.Callable[[_minBy__T, _minBy__T], int],
    ) -> Collector[_minBy__T, typing.Any, java.util.Optional[_minBy__T]]: ...
    _partitioningBy_0__T = typing.TypeVar("_partitioningBy_0__T")  # <T>
    _partitioningBy_1__T = typing.TypeVar("_partitioningBy_1__T")  # <T>
    _partitioningBy_1__D = typing.TypeVar("_partitioningBy_1__D")  # <D>
    _partitioningBy_1__A = typing.TypeVar("_partitioningBy_1__A")  # <A>
    @typing.overload
    @staticmethod
    def partitioningBy(
        predicate: java.util.function.Predicate[_partitioningBy_0__T]
        | typing.Callable[[_partitioningBy_0__T], bool],
    ) -> Collector[
        _partitioningBy_0__T,
        typing.Any,
        java.util.Map[bool, java.util.List[_partitioningBy_0__T]],
    ]: ...
    @typing.overload
    @staticmethod
    def partitioningBy(
        predicate: java.util.function.Predicate[_partitioningBy_1__T]
        | typing.Callable[[_partitioningBy_1__T], bool],
        collector: Collector[
            _partitioningBy_1__T, _partitioningBy_1__A, _partitioningBy_1__D
        ],
    ) -> Collector[
        _partitioningBy_1__T, typing.Any, java.util.Map[bool, _partitioningBy_1__D]
    ]: ...
    _reducing_0__T = typing.TypeVar("_reducing_0__T")  # <T>
    _reducing_1__T = typing.TypeVar("_reducing_1__T")  # <T>
    _reducing_1__U = typing.TypeVar("_reducing_1__U")  # <U>
    _reducing_2__T = typing.TypeVar("_reducing_2__T")  # <T>
    @typing.overload
    @staticmethod
    def reducing(
        t: _reducing_0__T,
        binaryOperator: java.util.function.BinaryOperator[_reducing_0__T]
        | typing.Callable,
    ) -> Collector[_reducing_0__T, typing.Any, _reducing_0__T]: ...
    @typing.overload
    @staticmethod
    def reducing(
        u: _reducing_1__U,
        function: java.util.function.Function[_reducing_1__T, _reducing_1__U]
        | typing.Callable[[_reducing_1__T], _reducing_1__U],
        binaryOperator: java.util.function.BinaryOperator[_reducing_1__U]
        | typing.Callable,
    ) -> Collector[_reducing_1__T, typing.Any, _reducing_1__U]: ...
    @typing.overload
    @staticmethod
    def reducing(
        binaryOperator: java.util.function.BinaryOperator[_reducing_2__T]
        | typing.Callable,
    ) -> Collector[_reducing_2__T, typing.Any, java.util.Optional[_reducing_2__T]]: ...
    _summarizingDouble__T = typing.TypeVar("_summarizingDouble__T")  # <T>
    @staticmethod
    def summarizingDouble(
        toDoubleFunction: java.util.function.ToDoubleFunction[_summarizingDouble__T]
        | typing.Callable[[_summarizingDouble__T], float],
    ) -> Collector[
        _summarizingDouble__T, typing.Any, java.util.DoubleSummaryStatistics
    ]: ...
    _summarizingInt__T = typing.TypeVar("_summarizingInt__T")  # <T>
    @staticmethod
    def summarizingInt(
        toIntFunction: java.util.function.ToIntFunction[_summarizingInt__T]
        | typing.Callable[[_summarizingInt__T], int],
    ) -> Collector[_summarizingInt__T, typing.Any, java.util.IntSummaryStatistics]: ...
    _summarizingLong__T = typing.TypeVar("_summarizingLong__T")  # <T>
    @staticmethod
    def summarizingLong(
        toLongFunction: java.util.function.ToLongFunction[_summarizingLong__T]
        | typing.Callable[[_summarizingLong__T], int],
    ) -> Collector[
        _summarizingLong__T, typing.Any, java.util.LongSummaryStatistics
    ]: ...
    _summingDouble__T = typing.TypeVar("_summingDouble__T")  # <T>
    @staticmethod
    def summingDouble(
        toDoubleFunction: java.util.function.ToDoubleFunction[_summingDouble__T]
        | typing.Callable[[_summingDouble__T], float],
    ) -> Collector[_summingDouble__T, typing.Any, float]: ...
    _summingInt__T = typing.TypeVar("_summingInt__T")  # <T>
    @staticmethod
    def summingInt(
        toIntFunction: java.util.function.ToIntFunction[_summingInt__T]
        | typing.Callable[[_summingInt__T], int],
    ) -> Collector[_summingInt__T, typing.Any, int]: ...
    _summingLong__T = typing.TypeVar("_summingLong__T")  # <T>
    @staticmethod
    def summingLong(
        toLongFunction: java.util.function.ToLongFunction[_summingLong__T]
        | typing.Callable[[_summingLong__T], int],
    ) -> Collector[_summingLong__T, typing.Any, int]: ...
    _teeing__T = typing.TypeVar("_teeing__T")  # <T>
    _teeing__R1 = typing.TypeVar("_teeing__R1")  # <R1>
    _teeing__R2 = typing.TypeVar("_teeing__R2")  # <R2>
    _teeing__R = typing.TypeVar("_teeing__R")  # <R>
    @staticmethod
    def teeing(
        collector: Collector[_teeing__T, typing.Any, _teeing__R1],
        collector2: Collector[_teeing__T, typing.Any, _teeing__R2],
        biFunction: java.util.function.BiFunction[_teeing__R1, _teeing__R2, _teeing__R]
        | typing.Callable[[_teeing__R1, _teeing__R2], _teeing__R],
    ) -> Collector[_teeing__T, typing.Any, _teeing__R]: ...
    _toCollection__T = typing.TypeVar("_toCollection__T")  # <T>
    _toCollection__C = typing.TypeVar(
        "_toCollection__C", bound=java.util.Collection
    )  # <C>
    @staticmethod
    def toCollection(
        supplier: java.util.function.Supplier[_toCollection__C]
        | typing.Callable[[], _toCollection__C],
    ) -> Collector[_toCollection__T, typing.Any, _toCollection__C]: ...
    _toConcurrentMap_0__T = typing.TypeVar("_toConcurrentMap_0__T")  # <T>
    _toConcurrentMap_0__K = typing.TypeVar("_toConcurrentMap_0__K")  # <K>
    _toConcurrentMap_0__U = typing.TypeVar("_toConcurrentMap_0__U")  # <U>
    _toConcurrentMap_1__T = typing.TypeVar("_toConcurrentMap_1__T")  # <T>
    _toConcurrentMap_1__K = typing.TypeVar("_toConcurrentMap_1__K")  # <K>
    _toConcurrentMap_1__U = typing.TypeVar("_toConcurrentMap_1__U")  # <U>
    _toConcurrentMap_2__T = typing.TypeVar("_toConcurrentMap_2__T")  # <T>
    _toConcurrentMap_2__K = typing.TypeVar("_toConcurrentMap_2__K")  # <K>
    _toConcurrentMap_2__U = typing.TypeVar("_toConcurrentMap_2__U")  # <U>
    _toConcurrentMap_2__M = typing.TypeVar(
        "_toConcurrentMap_2__M", bound=java.util.concurrent.ConcurrentMap
    )  # <M>
    @typing.overload
    @staticmethod
    def toConcurrentMap(
        function: java.util.function.Function[
            _toConcurrentMap_0__T, _toConcurrentMap_0__K
        ]
        | typing.Callable[[_toConcurrentMap_0__T], _toConcurrentMap_0__K],
        function2: java.util.function.Function[
            _toConcurrentMap_0__T, _toConcurrentMap_0__U
        ]
        | typing.Callable[[_toConcurrentMap_0__T], _toConcurrentMap_0__U],
    ) -> Collector[
        _toConcurrentMap_0__T,
        typing.Any,
        java.util.concurrent.ConcurrentMap[
            _toConcurrentMap_0__K, _toConcurrentMap_0__U
        ],
    ]: ...
    @typing.overload
    @staticmethod
    def toConcurrentMap(
        function: java.util.function.Function[
            _toConcurrentMap_1__T, _toConcurrentMap_1__K
        ]
        | typing.Callable[[_toConcurrentMap_1__T], _toConcurrentMap_1__K],
        function2: java.util.function.Function[
            _toConcurrentMap_1__T, _toConcurrentMap_1__U
        ]
        | typing.Callable[[_toConcurrentMap_1__T], _toConcurrentMap_1__U],
        binaryOperator: java.util.function.BinaryOperator[_toConcurrentMap_1__U]
        | typing.Callable,
    ) -> Collector[
        _toConcurrentMap_1__T,
        typing.Any,
        java.util.concurrent.ConcurrentMap[
            _toConcurrentMap_1__K, _toConcurrentMap_1__U
        ],
    ]: ...
    @typing.overload
    @staticmethod
    def toConcurrentMap(
        function: java.util.function.Function[
            _toConcurrentMap_2__T, _toConcurrentMap_2__K
        ]
        | typing.Callable[[_toConcurrentMap_2__T], _toConcurrentMap_2__K],
        function2: java.util.function.Function[
            _toConcurrentMap_2__T, _toConcurrentMap_2__U
        ]
        | typing.Callable[[_toConcurrentMap_2__T], _toConcurrentMap_2__U],
        binaryOperator: java.util.function.BinaryOperator[_toConcurrentMap_2__U]
        | typing.Callable,
        supplier: java.util.function.Supplier[_toConcurrentMap_2__M]
        | typing.Callable[[], _toConcurrentMap_2__M],
    ) -> Collector[_toConcurrentMap_2__T, typing.Any, _toConcurrentMap_2__M]: ...
    _toList__T = typing.TypeVar("_toList__T")  # <T>
    @staticmethod
    def toList() -> Collector[_toList__T, typing.Any, java.util.List[_toList__T]]: ...
    _toMap_0__T = typing.TypeVar("_toMap_0__T")  # <T>
    _toMap_0__K = typing.TypeVar("_toMap_0__K")  # <K>
    _toMap_0__U = typing.TypeVar("_toMap_0__U")  # <U>
    _toMap_1__T = typing.TypeVar("_toMap_1__T")  # <T>
    _toMap_1__K = typing.TypeVar("_toMap_1__K")  # <K>
    _toMap_1__U = typing.TypeVar("_toMap_1__U")  # <U>
    _toMap_2__T = typing.TypeVar("_toMap_2__T")  # <T>
    _toMap_2__K = typing.TypeVar("_toMap_2__K")  # <K>
    _toMap_2__U = typing.TypeVar("_toMap_2__U")  # <U>
    _toMap_2__M = typing.TypeVar("_toMap_2__M", bound=java.util.Map)  # <M>
    @typing.overload
    @staticmethod
    def toMap(
        function: java.util.function.Function[_toMap_0__T, _toMap_0__K]
        | typing.Callable[[_toMap_0__T], _toMap_0__K],
        function2: java.util.function.Function[_toMap_0__T, _toMap_0__U]
        | typing.Callable[[_toMap_0__T], _toMap_0__U],
    ) -> Collector[
        _toMap_0__T, typing.Any, java.util.Map[_toMap_0__K, _toMap_0__U]
    ]: ...
    @typing.overload
    @staticmethod
    def toMap(
        function: java.util.function.Function[_toMap_1__T, _toMap_1__K]
        | typing.Callable[[_toMap_1__T], _toMap_1__K],
        function2: java.util.function.Function[_toMap_1__T, _toMap_1__U]
        | typing.Callable[[_toMap_1__T], _toMap_1__U],
        binaryOperator: java.util.function.BinaryOperator[_toMap_1__U]
        | typing.Callable,
    ) -> Collector[
        _toMap_1__T, typing.Any, java.util.Map[_toMap_1__K, _toMap_1__U]
    ]: ...
    @typing.overload
    @staticmethod
    def toMap(
        function: java.util.function.Function[_toMap_2__T, _toMap_2__K]
        | typing.Callable[[_toMap_2__T], _toMap_2__K],
        function2: java.util.function.Function[_toMap_2__T, _toMap_2__U]
        | typing.Callable[[_toMap_2__T], _toMap_2__U],
        binaryOperator: java.util.function.BinaryOperator[_toMap_2__U]
        | typing.Callable,
        supplier: java.util.function.Supplier[_toMap_2__M]
        | typing.Callable[[], _toMap_2__M],
    ) -> Collector[_toMap_2__T, typing.Any, _toMap_2__M]: ...
    _toSet__T = typing.TypeVar("_toSet__T")  # <T>
    @staticmethod
    def toSet() -> Collector[_toSet__T, typing.Any, java.util.Set[_toSet__T]]: ...
    _toUnmodifiableList__T = typing.TypeVar("_toUnmodifiableList__T")  # <T>
    @staticmethod
    def toUnmodifiableList() -> Collector[
        _toUnmodifiableList__T, typing.Any, java.util.List[_toUnmodifiableList__T]
    ]: ...
    _toUnmodifiableMap_0__T = typing.TypeVar("_toUnmodifiableMap_0__T")  # <T>
    _toUnmodifiableMap_0__K = typing.TypeVar("_toUnmodifiableMap_0__K")  # <K>
    _toUnmodifiableMap_0__U = typing.TypeVar("_toUnmodifiableMap_0__U")  # <U>
    _toUnmodifiableMap_1__T = typing.TypeVar("_toUnmodifiableMap_1__T")  # <T>
    _toUnmodifiableMap_1__K = typing.TypeVar("_toUnmodifiableMap_1__K")  # <K>
    _toUnmodifiableMap_1__U = typing.TypeVar("_toUnmodifiableMap_1__U")  # <U>
    @typing.overload
    @staticmethod
    def toUnmodifiableMap(
        function: java.util.function.Function[
            _toUnmodifiableMap_0__T, _toUnmodifiableMap_0__K
        ]
        | typing.Callable[[_toUnmodifiableMap_0__T], _toUnmodifiableMap_0__K],
        function2: java.util.function.Function[
            _toUnmodifiableMap_0__T, _toUnmodifiableMap_0__U
        ]
        | typing.Callable[[_toUnmodifiableMap_0__T], _toUnmodifiableMap_0__U],
    ) -> Collector[
        _toUnmodifiableMap_0__T,
        typing.Any,
        java.util.Map[_toUnmodifiableMap_0__K, _toUnmodifiableMap_0__U],
    ]: ...
    @typing.overload
    @staticmethod
    def toUnmodifiableMap(
        function: java.util.function.Function[
            _toUnmodifiableMap_1__T, _toUnmodifiableMap_1__K
        ]
        | typing.Callable[[_toUnmodifiableMap_1__T], _toUnmodifiableMap_1__K],
        function2: java.util.function.Function[
            _toUnmodifiableMap_1__T, _toUnmodifiableMap_1__U
        ]
        | typing.Callable[[_toUnmodifiableMap_1__T], _toUnmodifiableMap_1__U],
        binaryOperator: java.util.function.BinaryOperator[_toUnmodifiableMap_1__U]
        | typing.Callable,
    ) -> Collector[
        _toUnmodifiableMap_1__T,
        typing.Any,
        java.util.Map[_toUnmodifiableMap_1__K, _toUnmodifiableMap_1__U],
    ]: ...
    _toUnmodifiableSet__T = typing.TypeVar("_toUnmodifiableSet__T")  # <T>
    @staticmethod
    def toUnmodifiableSet() -> Collector[
        _toUnmodifiableSet__T, typing.Any, java.util.Set[_toUnmodifiableSet__T]
    ]: ...

class Gatherers:
    _fold__T = typing.TypeVar("_fold__T")  # <T>
    _fold__R = typing.TypeVar("_fold__R")  # <R>
    @staticmethod
    def fold(
        supplier: java.util.function.Supplier[_fold__R] | typing.Callable[[], _fold__R],
        biFunction: java.util.function.BiFunction[_fold__R, _fold__T, _fold__R]
        | typing.Callable[[_fold__R, _fold__T], _fold__R],
    ) -> Gatherer[_fold__T, typing.Any, _fold__R]: ...
    _mapConcurrent__T = typing.TypeVar("_mapConcurrent__T")  # <T>
    _mapConcurrent__R = typing.TypeVar("_mapConcurrent__R")  # <R>
    @staticmethod
    def mapConcurrent(
        int: int,
        function: java.util.function.Function[_mapConcurrent__T, _mapConcurrent__R]
        | typing.Callable[[_mapConcurrent__T], _mapConcurrent__R],
    ) -> Gatherer[_mapConcurrent__T, typing.Any, _mapConcurrent__R]: ...
    _scan__T = typing.TypeVar("_scan__T")  # <T>
    _scan__R = typing.TypeVar("_scan__R")  # <R>
    @staticmethod
    def scan(
        supplier: java.util.function.Supplier[_scan__R] | typing.Callable[[], _scan__R],
        biFunction: java.util.function.BiFunction[_scan__R, _scan__T, _scan__R]
        | typing.Callable[[_scan__R, _scan__T], _scan__R],
    ) -> Gatherer[_scan__T, typing.Any, _scan__R]: ...
    _windowFixed__TR = typing.TypeVar("_windowFixed__TR")  # <TR>
    @staticmethod
    def windowFixed(
        int: int,
    ) -> Gatherer[_windowFixed__TR, typing.Any, java.util.List[_windowFixed__TR]]: ...
    _windowSliding__TR = typing.TypeVar("_windowSliding__TR")  # <TR>
    @staticmethod
    def windowSliding(
        int: int,
    ) -> Gatherer[
        _windowSliding__TR, typing.Any, java.util.List[_windowSliding__TR]
    ]: ...

class StreamSupport:
    @typing.overload
    @staticmethod
    def doubleStream(
        ofDouble: java.util.Spliterator.OfDouble, boolean: bool
    ) -> DoubleStream: ...
    @typing.overload
    @staticmethod
    def doubleStream(
        supplier: java.util.function.Supplier[java.util.Spliterator.OfDouble]
        | typing.Callable[[], java.util.Spliterator.OfDouble],
        int: int,
        boolean: bool,
    ) -> DoubleStream: ...
    @typing.overload
    @staticmethod
    def intStream(ofInt: java.util.Spliterator.OfInt, boolean: bool) -> IntStream: ...
    @typing.overload
    @staticmethod
    def intStream(
        supplier: java.util.function.Supplier[java.util.Spliterator.OfInt]
        | typing.Callable[[], java.util.Spliterator.OfInt],
        int: int,
        boolean: bool,
    ) -> IntStream: ...
    @typing.overload
    @staticmethod
    def longStream(
        ofLong: java.util.Spliterator.OfLong, boolean: bool
    ) -> LongStream: ...
    @typing.overload
    @staticmethod
    def longStream(
        supplier: java.util.function.Supplier[java.util.Spliterator.OfLong]
        | typing.Callable[[], java.util.Spliterator.OfLong],
        int: int,
        boolean: bool,
    ) -> LongStream: ...
    _stream_0__T = typing.TypeVar("_stream_0__T")  # <T>
    _stream_1__T = typing.TypeVar("_stream_1__T")  # <T>
    @typing.overload
    @staticmethod
    def stream(
        spliterator: java.util.Spliterator[_stream_0__T], boolean: bool
    ) -> Stream[_stream_0__T]: ...
    @typing.overload
    @staticmethod
    def stream(
        supplier: java.util.function.Supplier[java.util.Spliterator[_stream_1__T]]
        | typing.Callable[[], java.util.Spliterator[_stream_1__T]],
        int: int,
        boolean: bool,
    ) -> Stream[_stream_1__T]: ...

class DoubleStream(BaseStream[float, "DoubleStream"]):
    def allMatch(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> bool: ...
    def anyMatch(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> bool: ...
    def average(self) -> java.util.OptionalDouble: ...
    def boxed(self) -> Stream[float]: ...
    @staticmethod
    def builder() -> DoubleStream.Builder: ...
    _collect__R = typing.TypeVar("_collect__R")  # <R>
    def collect(
        self,
        supplier: java.util.function.Supplier[_collect__R]
        | typing.Callable[[], _collect__R],
        objDoubleConsumer: java.util.function.ObjDoubleConsumer[_collect__R]
        | typing.Callable[[_collect__R, float], None],
        biConsumer: java.util.function.BiConsumer[_collect__R, _collect__R]
        | typing.Callable[[_collect__R, _collect__R], None],
    ) -> _collect__R: ...
    @staticmethod
    def concat(
        doubleStream: DoubleStream, doubleStream2: DoubleStream
    ) -> DoubleStream: ...
    def count(self) -> int: ...
    def distinct(self) -> DoubleStream: ...
    def dropWhile(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> DoubleStream: ...
    @staticmethod
    def empty() -> DoubleStream: ...
    def filter(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> DoubleStream: ...
    def findAny(self) -> java.util.OptionalDouble: ...
    def findFirst(self) -> java.util.OptionalDouble: ...
    def flatMap(
        self,
        doubleFunction: java.util.function.DoubleFunction[DoubleStream]
        | typing.Callable[[float], DoubleStream],
    ) -> DoubleStream: ...
    def forEach(
        self, doubleConsumer: java.util.function.DoubleConsumer | typing.Callable
    ) -> None: ...
    def forEachOrdered(
        self, doubleConsumer: java.util.function.DoubleConsumer | typing.Callable
    ) -> None: ...
    @staticmethod
    def generate(
        doubleSupplier: java.util.function.DoubleSupplier | typing.Callable,
    ) -> DoubleStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        double: float,
        doublePredicate: java.util.function.DoublePredicate | typing.Callable,
        doubleUnaryOperator: java.util.function.DoubleUnaryOperator | typing.Callable,
    ) -> DoubleStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        double: float,
        doubleUnaryOperator: java.util.function.DoubleUnaryOperator | typing.Callable,
    ) -> DoubleStream: ...
    def iterator(self) -> java.util.PrimitiveIterator.OfDouble: ...
    def limit(self, long: int) -> DoubleStream: ...
    def map(
        self,
        doubleUnaryOperator: java.util.function.DoubleUnaryOperator | typing.Callable,
    ) -> DoubleStream: ...
    def mapMulti(
        self,
        doubleMapMultiConsumer: DoubleStream.DoubleMapMultiConsumer | typing.Callable,
    ) -> DoubleStream: ...
    def mapToInt(
        self,
        doubleToIntFunction: java.util.function.DoubleToIntFunction | typing.Callable,
    ) -> IntStream: ...
    def mapToLong(
        self,
        doubleToLongFunction: java.util.function.DoubleToLongFunction | typing.Callable,
    ) -> LongStream: ...
    _mapToObj__U = typing.TypeVar("_mapToObj__U")  # <U>
    def mapToObj(
        self,
        doubleFunction: java.util.function.DoubleFunction[_mapToObj__U]
        | typing.Callable[[float], _mapToObj__U],
    ) -> Stream[_mapToObj__U]: ...
    def max(self) -> java.util.OptionalDouble: ...
    def min(self) -> java.util.OptionalDouble: ...
    def noneMatch(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> bool: ...
    @typing.overload
    @staticmethod
    def of(double: float) -> DoubleStream: ...
    @typing.overload
    @staticmethod
    def of(*double: float) -> DoubleStream: ...
    def parallel(self) -> DoubleStream: ...
    def peek(
        self, doubleConsumer: java.util.function.DoubleConsumer | typing.Callable
    ) -> DoubleStream: ...
    @typing.overload
    def reduce(
        self,
        double: float,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> float: ...
    @typing.overload
    def reduce(
        self,
        doubleBinaryOperator: java.util.function.DoubleBinaryOperator | typing.Callable,
    ) -> java.util.OptionalDouble: ...
    def sequential(self) -> DoubleStream: ...
    def skip(self, long: int) -> DoubleStream: ...
    def sorted(self) -> DoubleStream: ...
    def spliterator(self) -> java.util.Spliterator.OfDouble: ...
    def sum(self) -> float: ...
    def summaryStatistics(self) -> java.util.DoubleSummaryStatistics: ...
    def takeWhile(
        self, doublePredicate: java.util.function.DoublePredicate | typing.Callable
    ) -> DoubleStream: ...
    def toArray(self) -> typing.MutableSequence[float]: ...
    class Builder(java.util.function.DoubleConsumer):
        def accept(self, double: float) -> None: ...
        def add(self, double: float) -> DoubleStream.Builder: ...
        def build(self) -> DoubleStream: ...

    class DoubleMapMultiConsumer:
        def accept(
            self,
            double: float,
            doubleConsumer: java.util.function.DoubleConsumer | typing.Callable,
        ) -> None: ...

class IntStream(BaseStream[int, "IntStream"]):
    def allMatch(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> bool: ...
    def anyMatch(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> bool: ...
    def asDoubleStream(self) -> DoubleStream: ...
    def asLongStream(self) -> LongStream: ...
    def average(self) -> java.util.OptionalDouble: ...
    def boxed(self) -> Stream[int]: ...
    @staticmethod
    def builder() -> IntStream.Builder: ...
    _collect__R = typing.TypeVar("_collect__R")  # <R>
    def collect(
        self,
        supplier: java.util.function.Supplier[_collect__R]
        | typing.Callable[[], _collect__R],
        objIntConsumer: java.util.function.ObjIntConsumer[_collect__R]
        | typing.Callable[[_collect__R, int], None],
        biConsumer: java.util.function.BiConsumer[_collect__R, _collect__R]
        | typing.Callable[[_collect__R, _collect__R], None],
    ) -> _collect__R: ...
    @staticmethod
    def concat(intStream: IntStream, intStream2: IntStream) -> IntStream: ...
    def count(self) -> int: ...
    def distinct(self) -> IntStream: ...
    def dropWhile(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> IntStream: ...
    @staticmethod
    def empty() -> IntStream: ...
    def filter(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> IntStream: ...
    def findAny(self) -> java.util.OptionalInt: ...
    def findFirst(self) -> java.util.OptionalInt: ...
    def flatMap(
        self,
        intFunction: java.util.function.IntFunction[IntStream]
        | typing.Callable[[int], IntStream],
    ) -> IntStream: ...
    def forEach(
        self, intConsumer: java.util.function.IntConsumer | typing.Callable
    ) -> None: ...
    def forEachOrdered(
        self, intConsumer: java.util.function.IntConsumer | typing.Callable
    ) -> None: ...
    @staticmethod
    def generate(
        intSupplier: java.util.function.IntSupplier | typing.Callable,
    ) -> IntStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        int: int,
        intPredicate: java.util.function.IntPredicate | typing.Callable,
        intUnaryOperator: java.util.function.IntUnaryOperator | typing.Callable,
    ) -> IntStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        int: int,
        intUnaryOperator: java.util.function.IntUnaryOperator | typing.Callable,
    ) -> IntStream: ...
    def iterator(self) -> java.util.PrimitiveIterator.OfInt: ...
    def limit(self, long: int) -> IntStream: ...
    def map(
        self, intUnaryOperator: java.util.function.IntUnaryOperator | typing.Callable
    ) -> IntStream: ...
    def mapMulti(
        self, intMapMultiConsumer: IntStream.IntMapMultiConsumer | typing.Callable
    ) -> IntStream: ...
    def mapToDouble(
        self,
        intToDoubleFunction: java.util.function.IntToDoubleFunction | typing.Callable,
    ) -> DoubleStream: ...
    def mapToLong(
        self, intToLongFunction: java.util.function.IntToLongFunction | typing.Callable
    ) -> LongStream: ...
    _mapToObj__U = typing.TypeVar("_mapToObj__U")  # <U>
    def mapToObj(
        self,
        intFunction: java.util.function.IntFunction[_mapToObj__U]
        | typing.Callable[[int], _mapToObj__U],
    ) -> Stream[_mapToObj__U]: ...
    def max(self) -> java.util.OptionalInt: ...
    def min(self) -> java.util.OptionalInt: ...
    def noneMatch(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> bool: ...
    @typing.overload
    @staticmethod
    def of(int: int) -> IntStream: ...
    @typing.overload
    @staticmethod
    def of(*int: int) -> IntStream: ...
    def parallel(self) -> IntStream: ...
    def peek(
        self, intConsumer: java.util.function.IntConsumer | typing.Callable
    ) -> IntStream: ...
    @staticmethod
    def range(int: int, int2: int) -> IntStream: ...
    @staticmethod
    def rangeClosed(int: int, int2: int) -> IntStream: ...
    @typing.overload
    def reduce(
        self,
        int: int,
        intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable,
    ) -> int: ...
    @typing.overload
    def reduce(
        self, intBinaryOperator: java.util.function.IntBinaryOperator | typing.Callable
    ) -> java.util.OptionalInt: ...
    def sequential(self) -> IntStream: ...
    def skip(self, long: int) -> IntStream: ...
    def sorted(self) -> IntStream: ...
    def spliterator(self) -> java.util.Spliterator.OfInt: ...
    def sum(self) -> int: ...
    def summaryStatistics(self) -> java.util.IntSummaryStatistics: ...
    def takeWhile(
        self, intPredicate: java.util.function.IntPredicate | typing.Callable
    ) -> IntStream: ...
    def toArray(self) -> typing.MutableSequence[int]: ...
    class Builder(java.util.function.IntConsumer):
        def accept(self, int: int) -> None: ...
        def add(self, int: int) -> IntStream.Builder: ...
        def build(self) -> IntStream: ...

    class IntMapMultiConsumer:
        def accept(
            self,
            int: int,
            intConsumer: java.util.function.IntConsumer | typing.Callable,
        ) -> None: ...

class LongStream(BaseStream[int, "LongStream"]):
    def allMatch(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> bool: ...
    def anyMatch(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> bool: ...
    def asDoubleStream(self) -> DoubleStream: ...
    def average(self) -> java.util.OptionalDouble: ...
    def boxed(self) -> Stream[int]: ...
    @staticmethod
    def builder() -> LongStream.Builder: ...
    _collect__R = typing.TypeVar("_collect__R")  # <R>
    def collect(
        self,
        supplier: java.util.function.Supplier[_collect__R]
        | typing.Callable[[], _collect__R],
        objLongConsumer: java.util.function.ObjLongConsumer[_collect__R]
        | typing.Callable[[_collect__R, int], None],
        biConsumer: java.util.function.BiConsumer[_collect__R, _collect__R]
        | typing.Callable[[_collect__R, _collect__R], None],
    ) -> _collect__R: ...
    @staticmethod
    def concat(longStream: LongStream, longStream2: LongStream) -> LongStream: ...
    def count(self) -> int: ...
    def distinct(self) -> LongStream: ...
    def dropWhile(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> LongStream: ...
    @staticmethod
    def empty() -> LongStream: ...
    def filter(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> LongStream: ...
    def findAny(self) -> java.util.OptionalLong: ...
    def findFirst(self) -> java.util.OptionalLong: ...
    def flatMap(
        self,
        longFunction: java.util.function.LongFunction[LongStream]
        | typing.Callable[[int], LongStream],
    ) -> LongStream: ...
    def forEach(
        self, longConsumer: java.util.function.LongConsumer | typing.Callable
    ) -> None: ...
    def forEachOrdered(
        self, longConsumer: java.util.function.LongConsumer | typing.Callable
    ) -> None: ...
    @staticmethod
    def generate(
        longSupplier: java.util.function.LongSupplier | typing.Callable,
    ) -> LongStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        long: int,
        longPredicate: java.util.function.LongPredicate | typing.Callable,
        longUnaryOperator: java.util.function.LongUnaryOperator | typing.Callable,
    ) -> LongStream: ...
    @typing.overload
    @staticmethod
    def iterate(
        long: int,
        longUnaryOperator: java.util.function.LongUnaryOperator | typing.Callable,
    ) -> LongStream: ...
    def iterator(self) -> java.util.PrimitiveIterator.OfLong: ...
    def limit(self, long: int) -> LongStream: ...
    def map(
        self, longUnaryOperator: java.util.function.LongUnaryOperator | typing.Callable
    ) -> LongStream: ...
    def mapMulti(
        self, longMapMultiConsumer: LongStream.LongMapMultiConsumer | typing.Callable
    ) -> LongStream: ...
    def mapToDouble(
        self,
        longToDoubleFunction: java.util.function.LongToDoubleFunction | typing.Callable,
    ) -> DoubleStream: ...
    def mapToInt(
        self, longToIntFunction: java.util.function.LongToIntFunction | typing.Callable
    ) -> IntStream: ...
    _mapToObj__U = typing.TypeVar("_mapToObj__U")  # <U>
    def mapToObj(
        self,
        longFunction: java.util.function.LongFunction[_mapToObj__U]
        | typing.Callable[[int], _mapToObj__U],
    ) -> Stream[_mapToObj__U]: ...
    def max(self) -> java.util.OptionalLong: ...
    def min(self) -> java.util.OptionalLong: ...
    def noneMatch(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> bool: ...
    @typing.overload
    @staticmethod
    def of(long: int) -> LongStream: ...
    @typing.overload
    @staticmethod
    def of(*long: int) -> LongStream: ...
    def parallel(self) -> LongStream: ...
    def peek(
        self, longConsumer: java.util.function.LongConsumer | typing.Callable
    ) -> LongStream: ...
    @staticmethod
    def range(long: int, long2: int) -> LongStream: ...
    @staticmethod
    def rangeClosed(long: int, long2: int) -> LongStream: ...
    @typing.overload
    def reduce(
        self,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> java.util.OptionalLong: ...
    @typing.overload
    def reduce(
        self,
        long: int,
        longBinaryOperator: java.util.function.LongBinaryOperator | typing.Callable,
    ) -> int: ...
    def sequential(self) -> LongStream: ...
    def skip(self, long: int) -> LongStream: ...
    def sorted(self) -> LongStream: ...
    def spliterator(self) -> java.util.Spliterator.OfLong: ...
    def sum(self) -> int: ...
    def summaryStatistics(self) -> java.util.LongSummaryStatistics: ...
    def takeWhile(
        self, longPredicate: java.util.function.LongPredicate | typing.Callable
    ) -> LongStream: ...
    def toArray(self) -> typing.MutableSequence[int]: ...
    class Builder(java.util.function.LongConsumer):
        def accept(self, long: int) -> None: ...
        def add(self, long: int) -> LongStream.Builder: ...
        def build(self) -> LongStream: ...

    class LongMapMultiConsumer:
        def accept(
            self,
            long: int,
            longConsumer: java.util.function.LongConsumer | typing.Callable,
        ) -> None: ...

_Stream__Builder__T = typing.TypeVar("_Stream__Builder__T")  # <T>
_Stream__T = typing.TypeVar("_Stream__T")  # <T>

class Stream(BaseStream[_Stream__T, "Stream"[_Stream__T]], typing.Generic[_Stream__T]):
    def allMatch(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> bool: ...
    def anyMatch(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> bool: ...
    _builder__T = typing.TypeVar("_builder__T")  # <T>
    @staticmethod
    def builder() -> Stream.Builder[_builder__T]: ...
    _collect_0__R = typing.TypeVar("_collect_0__R")  # <R>
    _collect_1__R = typing.TypeVar("_collect_1__R")  # <R>
    _collect_1__A = typing.TypeVar("_collect_1__A")  # <A>
    @typing.overload
    def collect(
        self,
        supplier: java.util.function.Supplier[_collect_0__R]
        | typing.Callable[[], _collect_0__R],
        biConsumer: java.util.function.BiConsumer[_collect_0__R, _Stream__T]
        | typing.Callable[[_collect_0__R, _Stream__T], None],
        biConsumer2: java.util.function.BiConsumer[_collect_0__R, _collect_0__R]
        | typing.Callable[[_collect_0__R, _collect_0__R], None],
    ) -> _collect_0__R: ...
    @typing.overload
    def collect(
        self, collector: Collector[_Stream__T, _collect_1__A, _collect_1__R]
    ) -> _collect_1__R: ...
    _concat__T = typing.TypeVar("_concat__T")  # <T>
    @staticmethod
    def concat(
        stream: Stream[_concat__T], stream2: Stream[_concat__T]
    ) -> Stream[_concat__T]: ...
    def count(self) -> int: ...
    def distinct(self) -> Stream[_Stream__T]: ...
    def dropWhile(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> Stream[_Stream__T]: ...
    _empty__T = typing.TypeVar("_empty__T")  # <T>
    @staticmethod
    def empty() -> Stream[_empty__T]: ...
    def filter(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> Stream[_Stream__T]: ...
    def findAny(self) -> java.util.Optional[_Stream__T]: ...
    def findFirst(self) -> java.util.Optional[_Stream__T]: ...
    _flatMap__R = typing.TypeVar("_flatMap__R")  # <R>
    def flatMap(
        self,
        function: java.util.function.Function[_Stream__T, Stream[_flatMap__R]]
        | typing.Callable[[_Stream__T], Stream[_flatMap__R]],
    ) -> Stream[_flatMap__R]: ...
    def flatMapToDouble(
        self,
        function: java.util.function.Function[_Stream__T, DoubleStream]
        | typing.Callable[[_Stream__T], DoubleStream],
    ) -> DoubleStream: ...
    def flatMapToInt(
        self,
        function: java.util.function.Function[_Stream__T, IntStream]
        | typing.Callable[[_Stream__T], IntStream],
    ) -> IntStream: ...
    def flatMapToLong(
        self,
        function: java.util.function.Function[_Stream__T, LongStream]
        | typing.Callable[[_Stream__T], LongStream],
    ) -> LongStream: ...
    def forEach(
        self,
        consumer: java.util.function.Consumer[_Stream__T]
        | typing.Callable[[_Stream__T], None],
    ) -> None: ...
    def forEachOrdered(
        self,
        consumer: java.util.function.Consumer[_Stream__T]
        | typing.Callable[[_Stream__T], None],
    ) -> None: ...
    _gather__R = typing.TypeVar("_gather__R")  # <R>
    def gather(
        self,
        gatherer: Gatherer[_Stream__T, typing.Any, _gather__R]
        | typing.Callable[[], Gatherer.Integrator[typing.Any, typing.Any, typing.Any]],
    ) -> Stream[_gather__R]: ...
    _generate__T = typing.TypeVar("_generate__T")  # <T>
    @staticmethod
    def generate(
        supplier: java.util.function.Supplier[_generate__T]
        | typing.Callable[[], _generate__T],
    ) -> Stream[_generate__T]: ...
    _iterate_0__T = typing.TypeVar("_iterate_0__T")  # <T>
    _iterate_1__T = typing.TypeVar("_iterate_1__T")  # <T>
    @typing.overload
    @staticmethod
    def iterate(
        t: _iterate_0__T,
        predicate: java.util.function.Predicate[_iterate_0__T]
        | typing.Callable[[_iterate_0__T], bool],
        unaryOperator: java.util.function.UnaryOperator[_iterate_0__T]
        | typing.Callable,
    ) -> Stream[_iterate_0__T]: ...
    @typing.overload
    @staticmethod
    def iterate(
        t: _iterate_1__T,
        unaryOperator: java.util.function.UnaryOperator[_iterate_1__T]
        | typing.Callable,
    ) -> Stream[_iterate_1__T]: ...
    def limit(self, long: int) -> Stream[_Stream__T]: ...
    _map__R = typing.TypeVar("_map__R")  # <R>
    def map(
        self,
        function: java.util.function.Function[_Stream__T, _map__R]
        | typing.Callable[[_Stream__T], _map__R],
    ) -> Stream[_map__R]: ...
    _mapMulti__R = typing.TypeVar("_mapMulti__R")  # <R>
    def mapMulti(
        self,
        biConsumer: java.util.function.BiConsumer[
            _Stream__T, java.util.function.Consumer[_mapMulti__R]
        ]
        | typing.Callable[
            [_Stream__T, java.util.function.Consumer[_mapMulti__R]], None
        ],
    ) -> Stream[_mapMulti__R]: ...
    def mapMultiToDouble(
        self,
        biConsumer: java.util.function.BiConsumer[
            _Stream__T, java.util.function.DoubleConsumer
        ]
        | typing.Callable[[_Stream__T, java.util.function.DoubleConsumer], None],
    ) -> DoubleStream: ...
    def mapMultiToInt(
        self,
        biConsumer: java.util.function.BiConsumer[
            _Stream__T, java.util.function.IntConsumer
        ]
        | typing.Callable[[_Stream__T, java.util.function.IntConsumer], None],
    ) -> IntStream: ...
    def mapMultiToLong(
        self,
        biConsumer: java.util.function.BiConsumer[
            _Stream__T, java.util.function.LongConsumer
        ]
        | typing.Callable[[_Stream__T, java.util.function.LongConsumer], None],
    ) -> LongStream: ...
    def mapToDouble(
        self,
        toDoubleFunction: java.util.function.ToDoubleFunction[_Stream__T]
        | typing.Callable[[_Stream__T], float],
    ) -> DoubleStream: ...
    def mapToInt(
        self,
        toIntFunction: java.util.function.ToIntFunction[_Stream__T]
        | typing.Callable[[_Stream__T], int],
    ) -> IntStream: ...
    def mapToLong(
        self,
        toLongFunction: java.util.function.ToLongFunction[_Stream__T]
        | typing.Callable[[_Stream__T], int],
    ) -> LongStream: ...
    def max(
        self,
        comparator: java.util.Comparator[_Stream__T]
        | typing.Callable[[_Stream__T, _Stream__T], int],
    ) -> java.util.Optional[_Stream__T]: ...
    def min(
        self,
        comparator: java.util.Comparator[_Stream__T]
        | typing.Callable[[_Stream__T, _Stream__T], int],
    ) -> java.util.Optional[_Stream__T]: ...
    def noneMatch(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> bool: ...
    _of_0__T = typing.TypeVar("_of_0__T")  # <T>
    _of_1__T = typing.TypeVar("_of_1__T")  # <T>
    @typing.overload
    @staticmethod
    def of(t: _of_0__T) -> Stream[_of_0__T]: ...
    @typing.overload
    @staticmethod
    def of(*t: _of_1__T) -> Stream[_of_1__T]: ...
    _ofNullable__T = typing.TypeVar("_ofNullable__T")  # <T>
    @staticmethod
    def ofNullable(t: _ofNullable__T) -> Stream[_ofNullable__T]: ...
    def peek(
        self,
        consumer: java.util.function.Consumer[_Stream__T]
        | typing.Callable[[_Stream__T], None],
    ) -> Stream[_Stream__T]: ...
    _reduce_0__U = typing.TypeVar("_reduce_0__U")  # <U>
    @typing.overload
    def reduce(
        self,
        u: _reduce_0__U,
        biFunction: java.util.function.BiFunction[
            _reduce_0__U, _Stream__T, _reduce_0__U
        ]
        | typing.Callable[[_reduce_0__U, _Stream__T], _reduce_0__U],
        binaryOperator: java.util.function.BinaryOperator[_reduce_0__U]
        | typing.Callable,
    ) -> _reduce_0__U: ...
    @typing.overload
    def reduce(
        self,
        t: _Stream__T,
        binaryOperator: java.util.function.BinaryOperator[_Stream__T] | typing.Callable,
    ) -> _Stream__T: ...
    @typing.overload
    def reduce(
        self,
        binaryOperator: java.util.function.BinaryOperator[_Stream__T] | typing.Callable,
    ) -> java.util.Optional[_Stream__T]: ...
    def skip(self, long: int) -> Stream[_Stream__T]: ...
    @typing.overload
    def sorted(self) -> Stream[_Stream__T]: ...
    @typing.overload
    def sorted(
        self,
        comparator: java.util.Comparator[_Stream__T]
        | typing.Callable[[_Stream__T, _Stream__T], int],
    ) -> Stream[_Stream__T]: ...
    def takeWhile(
        self,
        predicate: java.util.function.Predicate[_Stream__T]
        | typing.Callable[[_Stream__T], bool],
    ) -> Stream[_Stream__T]: ...
    _toArray_1__A = typing.TypeVar("_toArray_1__A")  # <A>
    @typing.overload
    def toArray(self) -> typing.MutableSequence[typing.Any]: ...
    @typing.overload
    def toArray(
        self,
        intFunction: java.util.function.IntFunction[list[_toArray_1__A] | jpype.JArray]
        | typing.Callable[[int], list[_toArray_1__A] | jpype.JArray],
    ) -> typing.MutableSequence[_toArray_1__A]: ...
    def toList(self) -> java.util.List[_Stream__T]: ...
    class Builder(
        java.util.function.Consumer[_Stream__Builder__T],
        typing.Generic[_Stream__Builder__T],
    ):
        def accept(self, t: _Stream__Builder__T) -> None: ...
        def add(
            self, t: _Stream__Builder__T
        ) -> Stream.Builder[_Stream__Builder__T]: ...
        def build(self) -> Stream[_Stream__Builder__T]: ...

_Gatherer__Downstream__T = typing.TypeVar("_Gatherer__Downstream__T")  # <T>
_Gatherer__Integrator__Greedy__A = typing.TypeVar(
    "_Gatherer__Integrator__Greedy__A"
)  # <A>
_Gatherer__Integrator__Greedy__T = typing.TypeVar(
    "_Gatherer__Integrator__Greedy__T"
)  # <T>
_Gatherer__Integrator__Greedy__R = typing.TypeVar(
    "_Gatherer__Integrator__Greedy__R"
)  # <R>
_Gatherer__Integrator__A = typing.TypeVar("_Gatherer__Integrator__A")  # <A>
_Gatherer__Integrator__T = typing.TypeVar("_Gatherer__Integrator__T")  # <T>
_Gatherer__Integrator__R = typing.TypeVar("_Gatherer__Integrator__R")  # <R>
_Gatherer__T = typing.TypeVar("_Gatherer__T")  # <T>
_Gatherer__A = typing.TypeVar("_Gatherer__A")  # <A>
_Gatherer__R = typing.TypeVar("_Gatherer__R")  # <R>

class Gatherer(typing.Generic[_Gatherer__T, _Gatherer__A, _Gatherer__R]):
    _andThen__RR = typing.TypeVar("_andThen__RR")  # <RR>
    def andThen(
        self,
        gatherer: Gatherer[_Gatherer__R, typing.Any, _andThen__RR]
        | typing.Callable[[], Gatherer.Integrator[typing.Any, typing.Any, typing.Any]],
    ) -> Gatherer[_Gatherer__T, typing.Any, _andThen__RR]: ...
    def combiner(self) -> java.util.function.BinaryOperator[_Gatherer__A]: ...
    _defaultCombiner__A = typing.TypeVar("_defaultCombiner__A")  # <A>
    @staticmethod
    def defaultCombiner() -> java.util.function.BinaryOperator[_defaultCombiner__A]: ...
    _defaultFinisher__A = typing.TypeVar("_defaultFinisher__A")  # <A>
    _defaultFinisher__R = typing.TypeVar("_defaultFinisher__R")  # <R>
    @staticmethod
    def defaultFinisher() -> java.util.function.BiConsumer[
        _defaultFinisher__A, Gatherer.Downstream[_defaultFinisher__R]
    ]: ...
    _defaultInitializer__A = typing.TypeVar("_defaultInitializer__A")  # <A>
    @staticmethod
    def defaultInitializer() -> java.util.function.Supplier[_defaultInitializer__A]: ...
    def finisher(
        self,
    ) -> java.util.function.BiConsumer[
        _Gatherer__A, Gatherer.Downstream[_Gatherer__R]
    ]: ...
    def initializer(self) -> java.util.function.Supplier[_Gatherer__A]: ...
    def integrator(
        self,
    ) -> Gatherer.Integrator[_Gatherer__A, _Gatherer__T, _Gatherer__R]: ...
    _of_0__T = typing.TypeVar("_of_0__T")  # <T>
    _of_0__A = typing.TypeVar("_of_0__A")  # <A>
    _of_0__R = typing.TypeVar("_of_0__R")  # <R>
    _of_1__T = typing.TypeVar("_of_1__T")  # <T>
    _of_1__R = typing.TypeVar("_of_1__R")  # <R>
    _of_2__T = typing.TypeVar("_of_2__T")  # <T>
    _of_2__R = typing.TypeVar("_of_2__R")  # <R>
    @typing.overload
    @staticmethod
    def of(
        supplier: java.util.function.Supplier[_of_0__A] | typing.Callable[[], _of_0__A],
        integrator: Gatherer.Integrator[_of_0__A, _of_0__T, _of_0__R]
        | typing.Callable[[_of_0__A, _of_0__T, Gatherer.Downstream[typing.Any]], bool],
        binaryOperator: java.util.function.BinaryOperator[_of_0__A] | typing.Callable,
        biConsumer: java.util.function.BiConsumer[
            _of_0__A, Gatherer.Downstream[_of_0__R] | typing.Callable[[_of_0__R], bool]
        ]
        | typing.Callable[
            [
                _of_0__A,
                Gatherer.Downstream[_of_0__R] | typing.Callable[[_of_0__R], bool],
            ],
            None,
        ],
    ) -> Gatherer[_of_0__T, _of_0__A, _of_0__R]: ...
    @typing.overload
    @staticmethod
    def of(
        integrator: Gatherer.Integrator[None, _of_1__T, _of_1__R]
        | typing.Callable[[None, _of_1__T, Gatherer.Downstream[typing.Any]], bool],
    ) -> Gatherer[_of_1__T, None, _of_1__R]: ...
    @typing.overload
    @staticmethod
    def of(
        integrator: Gatherer.Integrator[None, _of_2__T, _of_2__R]
        | typing.Callable[[None, _of_2__T, Gatherer.Downstream[typing.Any]], bool],
        biConsumer: java.util.function.BiConsumer[
            None, Gatherer.Downstream[_of_2__R] | typing.Callable[[_of_2__R], bool]
        ]
        | typing.Callable[
            [None, Gatherer.Downstream[_of_2__R] | typing.Callable[[_of_2__R], bool]],
            None,
        ],
    ) -> Gatherer[_of_2__T, None, _of_2__R]: ...
    _ofSequential_0__T = typing.TypeVar("_ofSequential_0__T")  # <T>
    _ofSequential_0__A = typing.TypeVar("_ofSequential_0__A")  # <A>
    _ofSequential_0__R = typing.TypeVar("_ofSequential_0__R")  # <R>
    _ofSequential_1__T = typing.TypeVar("_ofSequential_1__T")  # <T>
    _ofSequential_1__A = typing.TypeVar("_ofSequential_1__A")  # <A>
    _ofSequential_1__R = typing.TypeVar("_ofSequential_1__R")  # <R>
    _ofSequential_2__T = typing.TypeVar("_ofSequential_2__T")  # <T>
    _ofSequential_2__R = typing.TypeVar("_ofSequential_2__R")  # <R>
    _ofSequential_3__T = typing.TypeVar("_ofSequential_3__T")  # <T>
    _ofSequential_3__R = typing.TypeVar("_ofSequential_3__R")  # <R>
    @typing.overload
    @staticmethod
    def ofSequential(
        supplier: java.util.function.Supplier[_ofSequential_0__A]
        | typing.Callable[[], _ofSequential_0__A],
        integrator: Gatherer.Integrator[
            _ofSequential_0__A, _ofSequential_0__T, _ofSequential_0__R
        ]
        | typing.Callable[
            [_ofSequential_0__A, _ofSequential_0__T, Gatherer.Downstream[typing.Any]],
            bool,
        ],
    ) -> Gatherer[_ofSequential_0__T, _ofSequential_0__A, _ofSequential_0__R]: ...
    @typing.overload
    @staticmethod
    def ofSequential(
        supplier: java.util.function.Supplier[_ofSequential_1__A]
        | typing.Callable[[], _ofSequential_1__A],
        integrator: Gatherer.Integrator[
            _ofSequential_1__A, _ofSequential_1__T, _ofSequential_1__R
        ]
        | typing.Callable[
            [_ofSequential_1__A, _ofSequential_1__T, Gatherer.Downstream[typing.Any]],
            bool,
        ],
        biConsumer: java.util.function.BiConsumer[
            _ofSequential_1__A,
            Gatherer.Downstream[_ofSequential_1__R]
            | typing.Callable[[_ofSequential_1__R], bool],
        ]
        | typing.Callable[
            [
                _ofSequential_1__A,
                Gatherer.Downstream[_ofSequential_1__R]
                | typing.Callable[[_ofSequential_1__R], bool],
            ],
            None,
        ],
    ) -> Gatherer[_ofSequential_1__T, _ofSequential_1__A, _ofSequential_1__R]: ...
    @typing.overload
    @staticmethod
    def ofSequential(
        integrator: Gatherer.Integrator[None, _ofSequential_2__T, _ofSequential_2__R]
        | typing.Callable[
            [None, _ofSequential_2__T, Gatherer.Downstream[typing.Any]], bool
        ],
    ) -> Gatherer[_ofSequential_2__T, None, _ofSequential_2__R]: ...
    @typing.overload
    @staticmethod
    def ofSequential(
        integrator: Gatherer.Integrator[None, _ofSequential_3__T, _ofSequential_3__R]
        | typing.Callable[
            [None, _ofSequential_3__T, Gatherer.Downstream[typing.Any]], bool
        ],
        biConsumer: java.util.function.BiConsumer[
            None,
            Gatherer.Downstream[_ofSequential_3__R]
            | typing.Callable[[_ofSequential_3__R], bool],
        ]
        | typing.Callable[
            [
                None,
                Gatherer.Downstream[_ofSequential_3__R]
                | typing.Callable[[_ofSequential_3__R], bool],
            ],
            None,
        ],
    ) -> Gatherer[_ofSequential_3__T, None, _ofSequential_3__R]: ...
    class Downstream(typing.Generic[_Gatherer__Downstream__T]):
        def isRejecting(self) -> bool: ...
        def push(self, t: _Gatherer__Downstream__T) -> bool: ...

    class Integrator(
        typing.Generic[
            _Gatherer__Integrator__A, _Gatherer__Integrator__T, _Gatherer__Integrator__R
        ]
    ):
        def integrate(
            self,
            a: _Gatherer__Integrator__A,
            t: _Gatherer__Integrator__T,
            downstream: Gatherer.Downstream[_Gatherer__Integrator__R]
            | typing.Callable[[_Gatherer__Integrator__R], bool],
        ) -> bool: ...
        _of__A = typing.TypeVar("_of__A")  # <A>
        _of__T = typing.TypeVar("_of__T")  # <T>
        _of__R = typing.TypeVar("_of__R")  # <R>
        @staticmethod
        def of(
            integrator: Gatherer.Integrator[_of__A, _of__T, _of__R]
            | typing.Callable[[_of__A, _of__T, Gatherer.Downstream[typing.Any]], bool],
        ) -> Gatherer.Integrator[_of__A, _of__T, _of__R]: ...
        _ofGreedy__A = typing.TypeVar("_ofGreedy__A")  # <A>
        _ofGreedy__T = typing.TypeVar("_ofGreedy__T")  # <T>
        _ofGreedy__R = typing.TypeVar("_ofGreedy__R")  # <R>
        @staticmethod
        def ofGreedy(
            greedy: Gatherer.Integrator.Greedy[_ofGreedy__A, _ofGreedy__T, _ofGreedy__R]
            | typing.Callable,
        ) -> Gatherer.Integrator.Greedy[_ofGreedy__A, _ofGreedy__T, _ofGreedy__R]: ...
        class Greedy(
            java.util.stream.Gatherer.Integrator[
                _Gatherer__Integrator__Greedy__A,
                _Gatherer__Integrator__Greedy__T,
                _Gatherer__Integrator__Greedy__R,
            ],
            typing.Generic[
                _Gatherer__Integrator__Greedy__A,
                _Gatherer__Integrator__Greedy__T,
                _Gatherer__Integrator__Greedy__R,
            ],
        ): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.util.stream")``.

    BaseStream: type[BaseStream]
    Collector: type[Collector]
    Collectors: type[Collectors]
    DoubleStream: type[DoubleStream]
    Gatherer: type[Gatherer]
    Gatherers: type[Gatherers]
    IntStream: type[IntStream]
    LongStream: type[LongStream]
    Stream: type[Stream]
    StreamSupport: type[StreamSupport]
