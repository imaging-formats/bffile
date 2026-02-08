import typing
from typing import Protocol

import java.lang
import jpype
import loci.common
import loci.poi.poifs.filesystem

class POIFSReader:
    def __init__(self): ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def read(
        self, randomAccessInputStream: loci.common.RandomAccessInputStream, int: int
    ) -> None: ...
    @typing.overload
    def registerListener(
        self, pOIFSReaderListener: POIFSReaderListener | typing.Callable
    ) -> None: ...
    @typing.overload
    def registerListener(
        self,
        pOIFSReaderListener: POIFSReaderListener | typing.Callable,
        string: java.lang.String | str,
    ) -> None: ...
    @typing.overload
    def registerListener(
        self,
        pOIFSReaderListener: POIFSReaderListener | typing.Callable,
        pOIFSDocumentPath: loci.poi.poifs.filesystem.POIFSDocumentPath,
        string: java.lang.String | str,
    ) -> None: ...

class POIFSReaderEvent:
    def getName(self) -> java.lang.String: ...
    def getPath(self) -> loci.poi.poifs.filesystem.POIFSDocumentPath: ...
    def getStream(self) -> loci.poi.poifs.filesystem.DocumentInputStream: ...

class POIFSReaderListener:
    def processPOIFSReaderEvent(self, pOIFSReaderEvent: POIFSReaderEvent) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.poifs.eventfilesystem")``.

    POIFSReader: type[POIFSReader]
    POIFSReaderEvent: type[POIFSReaderEvent]
    POIFSReaderListener: type[POIFSReaderListener]
