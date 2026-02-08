import typing
from typing import Protocol

import jpype

class IImageScaler:
    def downsample(
        self,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        double: float,
        int3: int,
        boolean: bool,
        boolean2: bool,
        int4: int,
        boolean3: bool,
    ) -> typing.MutableSequence[int]: ...

class SimpleImageScaler(IImageScaler):
    def __init__(self): ...
    def downsample(
        self,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        double: float,
        int3: int,
        boolean: bool,
        boolean2: bool,
        int4: int,
        boolean3: bool,
    ) -> typing.MutableSequence[int]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.common.image")``.

    IImageScaler: type[IImageScaler]
    SimpleImageScaler: type[SimpleImageScaler]
