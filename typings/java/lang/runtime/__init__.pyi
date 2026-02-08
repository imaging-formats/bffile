import typing
from typing import Protocol

import java.lang
import java.lang.invoke

class ExactConversionsSupport:
    @staticmethod
    def isDoubleToByteExact(double: float) -> bool: ...
    @staticmethod
    def isDoubleToCharExact(double: float) -> bool: ...
    @staticmethod
    def isDoubleToFloatExact(double: float) -> bool: ...
    @staticmethod
    def isDoubleToIntExact(double: float) -> bool: ...
    @staticmethod
    def isDoubleToLongExact(double: float) -> bool: ...
    @staticmethod
    def isDoubleToShortExact(double: float) -> bool: ...
    @staticmethod
    def isFloatToByteExact(float: float) -> bool: ...
    @staticmethod
    def isFloatToCharExact(float: float) -> bool: ...
    @staticmethod
    def isFloatToIntExact(float: float) -> bool: ...
    @staticmethod
    def isFloatToLongExact(float: float) -> bool: ...
    @staticmethod
    def isFloatToShortExact(float: float) -> bool: ...
    @staticmethod
    def isIntToByteExact(int: int) -> bool: ...
    @staticmethod
    def isIntToCharExact(int: int) -> bool: ...
    @staticmethod
    def isIntToFloatExact(int: int) -> bool: ...
    @staticmethod
    def isIntToShortExact(int: int) -> bool: ...
    @staticmethod
    def isLongToByteExact(long: int) -> bool: ...
    @staticmethod
    def isLongToCharExact(long: int) -> bool: ...
    @staticmethod
    def isLongToDoubleExact(long: int) -> bool: ...
    @staticmethod
    def isLongToFloatExact(long: int) -> bool: ...
    @staticmethod
    def isLongToIntExact(long: int) -> bool: ...
    @staticmethod
    def isLongToShortExact(long: int) -> bool: ...

class ObjectMethods:
    @staticmethod
    def bootstrap(
        lookup: java.lang.invoke.MethodHandles.Lookup,
        string: java.lang.String | str,
        typeDescriptor: java.lang.invoke.TypeDescriptor | typing.Callable,
        class_: type[typing.Any],
        string2: java.lang.String | str,
        *methodHandle: java.lang.invoke.MethodHandle,
    ) -> typing.Any: ...

class SwitchBootstraps:
    @staticmethod
    def enumSwitch(
        lookup: java.lang.invoke.MethodHandles.Lookup,
        string: java.lang.String | str,
        methodType: java.lang.invoke.MethodType,
        *object: typing.Any,
    ) -> java.lang.invoke.CallSite: ...
    @staticmethod
    def typeSwitch(
        lookup: java.lang.invoke.MethodHandles.Lookup,
        string: java.lang.String | str,
        methodType: java.lang.invoke.MethodType,
        *object: typing.Any,
    ) -> java.lang.invoke.CallSite: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.lang.runtime")``.

    ExactConversionsSupport: type[ExactConversionsSupport]
    ObjectMethods: type[ObjectMethods]
    SwitchBootstraps: type[SwitchBootstraps]
