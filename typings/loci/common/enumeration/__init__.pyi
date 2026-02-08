import typing
from typing import Protocol

import java.lang

class CodedEnum:
    def getCode(self) -> int: ...

class EnumException(java.lang.RuntimeException):
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

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.common.enumeration")``.

    CodedEnum: type[CodedEnum]
    EnumException: type[EnumException]
