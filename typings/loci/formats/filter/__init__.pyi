import typing
from typing import Protocol

import java.lang
import java.util
import jpype
import ucar.nc2.filter

class LZ4Filter(ucar.nc2.filter.Filter):
    def __init__(
        self,
        map: java.util.Map[java.lang.String | str, typing.Any]
        | typing.Mapping[java.lang.String | str, typing.Any],
    ): ...
    def decode(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    def encode(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    def getId(self) -> int: ...
    def getName(self) -> java.lang.String: ...
    class LZ4FilterProvider(ucar.nc2.filter.FilterProvider):
        def __init__(self): ...
        def create(
            self,
            map: java.util.Map[java.lang.String | str, typing.Any]
            | typing.Mapping[java.lang.String | str, typing.Any],
        ) -> ucar.nc2.filter.Filter: ...
        def getId(self) -> int: ...
        def getName(self) -> java.lang.String: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.formats.filter")``.

    LZ4Filter: type[LZ4Filter]
