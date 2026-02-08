import typing
from typing import Protocol

import java.util

_BiConsumer__T = typing.TypeVar("_BiConsumer__T")  # <T>
_BiConsumer__U = typing.TypeVar("_BiConsumer__U")  # <U>

class BiConsumer(typing.Generic[_BiConsumer__T, _BiConsumer__U]):
    def accept(self, t: _BiConsumer__T, u: _BiConsumer__U) -> None: ...
    def andThen(
        self,
        biConsumer: BiConsumer[_BiConsumer__T, _BiConsumer__U]
        | typing.Callable[[_BiConsumer__T, _BiConsumer__U], None],
    ) -> BiConsumer[_BiConsumer__T, _BiConsumer__U]: ...

_BiFunction__T = typing.TypeVar("_BiFunction__T")  # <T>
_BiFunction__U = typing.TypeVar("_BiFunction__U")  # <U>
_BiFunction__R = typing.TypeVar("_BiFunction__R")  # <R>

class BiFunction(typing.Generic[_BiFunction__T, _BiFunction__U, _BiFunction__R]):
    _andThen__V = typing.TypeVar("_andThen__V")  # <V>
    def andThen(
        self,
        function: Function[_BiFunction__R, _andThen__V]
        | typing.Callable[[_BiFunction__R], _andThen__V],
    ) -> BiFunction[_BiFunction__T, _BiFunction__U, _andThen__V]: ...
    def apply(self, t: _BiFunction__T, u: _BiFunction__U) -> _BiFunction__R: ...

_BiPredicate__T = typing.TypeVar("_BiPredicate__T")  # <T>
_BiPredicate__U = typing.TypeVar("_BiPredicate__U")  # <U>

class BiPredicate(typing.Generic[_BiPredicate__T, _BiPredicate__U]):
    def and_(
        self,
        biPredicate: BiPredicate[_BiPredicate__T, _BiPredicate__U]
        | typing.Callable[[_BiPredicate__T, _BiPredicate__U], bool],
    ) -> BiPredicate[_BiPredicate__T, _BiPredicate__U]: ...
    def negate(self) -> BiPredicate[_BiPredicate__T, _BiPredicate__U]: ...
    def or_(
        self,
        biPredicate: BiPredicate[_BiPredicate__T, _BiPredicate__U]
        | typing.Callable[[_BiPredicate__T, _BiPredicate__U], bool],
    ) -> BiPredicate[_BiPredicate__T, _BiPredicate__U]: ...
    def test(self, t: _BiPredicate__T, u: _BiPredicate__U) -> bool: ...

class BooleanSupplier:
    def getAsBoolean(self) -> bool: ...

_Consumer__T = typing.TypeVar("_Consumer__T")  # <T>

class Consumer(typing.Generic[_Consumer__T]):
    def accept(self, t: _Consumer__T) -> None: ...
    def andThen(
        self, consumer: Consumer[_Consumer__T] | typing.Callable[[_Consumer__T], None]
    ) -> Consumer[_Consumer__T]: ...

class DoubleBinaryOperator:
    def applyAsDouble(self, double: float, double2: float) -> float: ...

class DoubleConsumer:
    def accept(self, double: float) -> None: ...
    def andThen(
        self, doubleConsumer: DoubleConsumer | typing.Callable
    ) -> DoubleConsumer: ...

_DoubleFunction__R = typing.TypeVar("_DoubleFunction__R")  # <R>

class DoubleFunction(typing.Generic[_DoubleFunction__R]):
    def apply(self, double: float) -> _DoubleFunction__R: ...

class DoublePredicate:
    def and_(
        self, doublePredicate: DoublePredicate | typing.Callable
    ) -> DoublePredicate: ...
    def negate(self) -> DoublePredicate: ...
    def or_(
        self, doublePredicate: DoublePredicate | typing.Callable
    ) -> DoublePredicate: ...
    def test(self, double: float) -> bool: ...

class DoubleSupplier:
    def getAsDouble(self) -> float: ...

class DoubleToIntFunction:
    def applyAsInt(self, double: float) -> int: ...

class DoubleToLongFunction:
    def applyAsLong(self, double: float) -> int: ...

class DoubleUnaryOperator:
    def andThen(
        self, doubleUnaryOperator: DoubleUnaryOperator | typing.Callable
    ) -> DoubleUnaryOperator: ...
    def applyAsDouble(self, double: float) -> float: ...
    def compose(
        self, doubleUnaryOperator: DoubleUnaryOperator | typing.Callable
    ) -> DoubleUnaryOperator: ...
    @staticmethod
    def identity() -> DoubleUnaryOperator: ...

_Function__T = typing.TypeVar("_Function__T")  # <T>
_Function__R = typing.TypeVar("_Function__R")  # <R>

class Function(typing.Generic[_Function__T, _Function__R]):
    _andThen__V = typing.TypeVar("_andThen__V")  # <V>
    def andThen(
        self,
        function: Function[_Function__R, _andThen__V]
        | typing.Callable[[_Function__R], _andThen__V],
    ) -> Function[_Function__T, _andThen__V]: ...
    def apply(self, t: _Function__T) -> _Function__R: ...
    _compose__V = typing.TypeVar("_compose__V")  # <V>
    def compose(
        self,
        function: Function[_compose__V, _Function__T]
        | typing.Callable[[_compose__V], _Function__T],
    ) -> Function[_compose__V, _Function__R]: ...
    _identity__T = typing.TypeVar("_identity__T")  # <T>
    @staticmethod
    def identity() -> Function[_identity__T, _identity__T]: ...

class IntBinaryOperator:
    def applyAsInt(self, int: int, int2: int) -> int: ...

class IntConsumer:
    def accept(self, int: int) -> None: ...
    def andThen(self, intConsumer: IntConsumer | typing.Callable) -> IntConsumer: ...

_IntFunction__R = typing.TypeVar("_IntFunction__R")  # <R>

class IntFunction(typing.Generic[_IntFunction__R]):
    def apply(self, int: int) -> _IntFunction__R: ...

class IntPredicate:
    def and_(self, intPredicate: IntPredicate | typing.Callable) -> IntPredicate: ...
    def negate(self) -> IntPredicate: ...
    def or_(self, intPredicate: IntPredicate | typing.Callable) -> IntPredicate: ...
    def test(self, int: int) -> bool: ...

class IntSupplier:
    def getAsInt(self) -> int: ...

class IntToDoubleFunction:
    def applyAsDouble(self, int: int) -> float: ...

class IntToLongFunction:
    def applyAsLong(self, int: int) -> int: ...

class IntUnaryOperator:
    def andThen(
        self, intUnaryOperator: IntUnaryOperator | typing.Callable
    ) -> IntUnaryOperator: ...
    def applyAsInt(self, int: int) -> int: ...
    def compose(
        self, intUnaryOperator: IntUnaryOperator | typing.Callable
    ) -> IntUnaryOperator: ...
    @staticmethod
    def identity() -> IntUnaryOperator: ...

class LongBinaryOperator:
    def applyAsLong(self, long: int, long2: int) -> int: ...

class LongConsumer:
    def accept(self, long: int) -> None: ...
    def andThen(self, longConsumer: LongConsumer | typing.Callable) -> LongConsumer: ...

_LongFunction__R = typing.TypeVar("_LongFunction__R")  # <R>

class LongFunction(typing.Generic[_LongFunction__R]):
    def apply(self, long: int) -> _LongFunction__R: ...

class LongPredicate:
    def and_(self, longPredicate: LongPredicate | typing.Callable) -> LongPredicate: ...
    def negate(self) -> LongPredicate: ...
    def or_(self, longPredicate: LongPredicate | typing.Callable) -> LongPredicate: ...
    def test(self, long: int) -> bool: ...

class LongSupplier:
    def getAsLong(self) -> int: ...

class LongToDoubleFunction:
    def applyAsDouble(self, long: int) -> float: ...

class LongToIntFunction:
    def applyAsInt(self, long: int) -> int: ...

class LongUnaryOperator:
    def andThen(
        self, longUnaryOperator: LongUnaryOperator | typing.Callable
    ) -> LongUnaryOperator: ...
    def applyAsLong(self, long: int) -> int: ...
    def compose(
        self, longUnaryOperator: LongUnaryOperator | typing.Callable
    ) -> LongUnaryOperator: ...
    @staticmethod
    def identity() -> LongUnaryOperator: ...

_ObjDoubleConsumer__T = typing.TypeVar("_ObjDoubleConsumer__T")  # <T>

class ObjDoubleConsumer(typing.Generic[_ObjDoubleConsumer__T]):
    def accept(self, t: _ObjDoubleConsumer__T, double: float) -> None: ...

_ObjIntConsumer__T = typing.TypeVar("_ObjIntConsumer__T")  # <T>

class ObjIntConsumer(typing.Generic[_ObjIntConsumer__T]):
    def accept(self, t: _ObjIntConsumer__T, int: int) -> None: ...

_ObjLongConsumer__T = typing.TypeVar("_ObjLongConsumer__T")  # <T>

class ObjLongConsumer(typing.Generic[_ObjLongConsumer__T]):
    def accept(self, t: _ObjLongConsumer__T, long: int) -> None: ...

_Predicate__T = typing.TypeVar("_Predicate__T")  # <T>

class Predicate(typing.Generic[_Predicate__T]):
    def and_(
        self,
        predicate: Predicate[_Predicate__T] | typing.Callable[[_Predicate__T], bool],
    ) -> Predicate[_Predicate__T]: ...
    _isEqual__T = typing.TypeVar("_isEqual__T")  # <T>
    @staticmethod
    def isEqual(object: typing.Any) -> Predicate[_isEqual__T]: ...
    def negate(self) -> Predicate[_Predicate__T]: ...
    _not___T = typing.TypeVar("_not___T")  # <T>
    @staticmethod
    def not_(
        predicate: Predicate[_not___T] | typing.Callable[[_not___T], bool],
    ) -> Predicate[_not___T]: ...
    def or_(
        self,
        predicate: Predicate[_Predicate__T] | typing.Callable[[_Predicate__T], bool],
    ) -> Predicate[_Predicate__T]: ...
    def test(self, t: _Predicate__T) -> bool: ...

_Supplier__T = typing.TypeVar("_Supplier__T")  # <T>

class Supplier(typing.Generic[_Supplier__T]):
    def get(self) -> _Supplier__T: ...

_ToDoubleBiFunction__T = typing.TypeVar("_ToDoubleBiFunction__T")  # <T>
_ToDoubleBiFunction__U = typing.TypeVar("_ToDoubleBiFunction__U")  # <U>

class ToDoubleBiFunction(
    typing.Generic[_ToDoubleBiFunction__T, _ToDoubleBiFunction__U]
):
    def applyAsDouble(
        self, t: _ToDoubleBiFunction__T, u: _ToDoubleBiFunction__U
    ) -> float: ...

_ToDoubleFunction__T = typing.TypeVar("_ToDoubleFunction__T")  # <T>

class ToDoubleFunction(typing.Generic[_ToDoubleFunction__T]):
    def applyAsDouble(self, t: _ToDoubleFunction__T) -> float: ...

_ToIntBiFunction__T = typing.TypeVar("_ToIntBiFunction__T")  # <T>
_ToIntBiFunction__U = typing.TypeVar("_ToIntBiFunction__U")  # <U>

class ToIntBiFunction(typing.Generic[_ToIntBiFunction__T, _ToIntBiFunction__U]):
    def applyAsInt(self, t: _ToIntBiFunction__T, u: _ToIntBiFunction__U) -> int: ...

_ToIntFunction__T = typing.TypeVar("_ToIntFunction__T")  # <T>

class ToIntFunction(typing.Generic[_ToIntFunction__T]):
    def applyAsInt(self, t: _ToIntFunction__T) -> int: ...

_ToLongBiFunction__T = typing.TypeVar("_ToLongBiFunction__T")  # <T>
_ToLongBiFunction__U = typing.TypeVar("_ToLongBiFunction__U")  # <U>

class ToLongBiFunction(typing.Generic[_ToLongBiFunction__T, _ToLongBiFunction__U]):
    def applyAsLong(self, t: _ToLongBiFunction__T, u: _ToLongBiFunction__U) -> int: ...

_ToLongFunction__T = typing.TypeVar("_ToLongFunction__T")  # <T>

class ToLongFunction(typing.Generic[_ToLongFunction__T]):
    def applyAsLong(self, t: _ToLongFunction__T) -> int: ...

_BinaryOperator__T = typing.TypeVar("_BinaryOperator__T")  # <T>

class BinaryOperator(
    BiFunction[_BinaryOperator__T, _BinaryOperator__T, _BinaryOperator__T],
    typing.Generic[_BinaryOperator__T],
):
    _maxBy__T = typing.TypeVar("_maxBy__T")  # <T>
    @staticmethod
    def maxBy(
        comparator: java.util.Comparator[_maxBy__T]
        | typing.Callable[[_maxBy__T, _maxBy__T], int],
    ) -> BinaryOperator[_maxBy__T]: ...
    _minBy__T = typing.TypeVar("_minBy__T")  # <T>
    @staticmethod
    def minBy(
        comparator: java.util.Comparator[_minBy__T]
        | typing.Callable[[_minBy__T, _minBy__T], int],
    ) -> BinaryOperator[_minBy__T]: ...

_UnaryOperator__T = typing.TypeVar("_UnaryOperator__T")  # <T>

class UnaryOperator(
    Function[_UnaryOperator__T, _UnaryOperator__T], typing.Generic[_UnaryOperator__T]
):
    _identity__T = typing.TypeVar("_identity__T")  # <T>
    @staticmethod
    def identity() -> UnaryOperator[_identity__T]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.util.function")``.

    BiConsumer: type[BiConsumer]
    BiFunction: type[BiFunction]
    BiPredicate: type[BiPredicate]
    BinaryOperator: type[BinaryOperator]
    BooleanSupplier: type[BooleanSupplier]
    Consumer: type[Consumer]
    DoubleBinaryOperator: type[DoubleBinaryOperator]
    DoubleConsumer: type[DoubleConsumer]
    DoubleFunction: type[DoubleFunction]
    DoublePredicate: type[DoublePredicate]
    DoubleSupplier: type[DoubleSupplier]
    DoubleToIntFunction: type[DoubleToIntFunction]
    DoubleToLongFunction: type[DoubleToLongFunction]
    DoubleUnaryOperator: type[DoubleUnaryOperator]
    Function: type[Function]
    IntBinaryOperator: type[IntBinaryOperator]
    IntConsumer: type[IntConsumer]
    IntFunction: type[IntFunction]
    IntPredicate: type[IntPredicate]
    IntSupplier: type[IntSupplier]
    IntToDoubleFunction: type[IntToDoubleFunction]
    IntToLongFunction: type[IntToLongFunction]
    IntUnaryOperator: type[IntUnaryOperator]
    LongBinaryOperator: type[LongBinaryOperator]
    LongConsumer: type[LongConsumer]
    LongFunction: type[LongFunction]
    LongPredicate: type[LongPredicate]
    LongSupplier: type[LongSupplier]
    LongToDoubleFunction: type[LongToDoubleFunction]
    LongToIntFunction: type[LongToIntFunction]
    LongUnaryOperator: type[LongUnaryOperator]
    ObjDoubleConsumer: type[ObjDoubleConsumer]
    ObjIntConsumer: type[ObjIntConsumer]
    ObjLongConsumer: type[ObjLongConsumer]
    Predicate: type[Predicate]
    Supplier: type[Supplier]
    ToDoubleBiFunction: type[ToDoubleBiFunction]
    ToDoubleFunction: type[ToDoubleFunction]
    ToIntBiFunction: type[ToIntBiFunction]
    ToIntFunction: type[ToIntFunction]
    ToLongBiFunction: type[ToLongBiFunction]
    ToLongFunction: type[ToLongFunction]
    UnaryOperator: type[UnaryOperator]
