import typing
from typing import Protocol

import java.lang
import java.util
import jpype

class POIFSViewEngine:
    def __init__(self): ...
    @staticmethod
    def inspectViewable(
        object: typing.Any, boolean: bool, int: int, string: java.lang.String | str
    ) -> java.util.List: ...

class POIFSViewable:
    def getShortDescription(self) -> java.lang.String: ...
    def getViewableArray(self) -> typing.MutableSequence[typing.Any]: ...
    def getViewableIterator(self) -> java.util.Iterator: ...
    def preferArray(self) -> bool: ...

class POIFSViewer:
    def __init__(self): ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.poifs.dev")``.

    POIFSViewEngine: type[POIFSViewEngine]
    POIFSViewable: type[POIFSViewable]
    POIFSViewer: type[POIFSViewer]
