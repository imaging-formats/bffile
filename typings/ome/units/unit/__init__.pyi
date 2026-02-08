import typing
from typing import Protocol

import _jpype
import java.lang
import ome.units.quantity

_Unit__Q = typing.TypeVar("_Unit__Q", bound=ome.units.quantity.Quantity)  # <Q>

class Unit(typing.Generic[_Unit__Q]):
    _CreateBaseUnit__Q = typing.TypeVar(
        "_CreateBaseUnit__Q", bound=ome.units.quantity.Quantity
    )  # <Q>
    @staticmethod
    def CreateBaseUnit(
        string: java.lang.String | str, string2: java.lang.String | str
    ) -> Unit[_CreateBaseUnit__Q]: ...
    @typing.overload
    def add(self, double: float) -> Unit[_Unit__Q]: ...
    @typing.overload
    def add(self, integer: int) -> Unit[_Unit__Q]: ...
    def convertValue(
        self,
        number: _jpype._JNumberLong
        | _jpype._JNumberFloat
        | typing.SupportsIndex
        | typing.SupportsFloat,
        unit: Unit[_Unit__Q],
    ) -> float: ...
    @typing.overload
    def divide(self, double: float) -> Unit[_Unit__Q]: ...
    @typing.overload
    def divide(self, integer: int) -> Unit[_Unit__Q]: ...
    def getOffset(self) -> float: ...
    def getScaleFactor(self) -> float: ...
    def getSymbol(self) -> java.lang.String: ...
    def isConvertible(self, unit: Unit[_Unit__Q]) -> bool: ...
    @typing.overload
    def multiply(self, double: float) -> Unit[_Unit__Q]: ...
    @typing.overload
    def multiply(self, integer: int) -> Unit[_Unit__Q]: ...
    def prefixSymbol(self, string: java.lang.String | str) -> Unit[_Unit__Q]: ...
    def setSymbol(self, string: java.lang.String | str) -> Unit[_Unit__Q]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.units.unit")``.

    Unit: type[Unit]
