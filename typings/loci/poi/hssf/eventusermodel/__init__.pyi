import typing
from typing import Protocol

import java.io
import java.lang
import loci.poi.hssf.record
import loci.poi.poifs.filesystem

class HSSFEventFactory:
    def __init__(self): ...
    def abortableProcessEvents(
        self, hSSFRequest: HSSFRequest, inputStream: java.io.InputStream
    ) -> int: ...
    def abortableProcessWorkbookEvents(
        self,
        hSSFRequest: HSSFRequest,
        pOIFSFileSystem: loci.poi.poifs.filesystem.POIFSFileSystem,
    ) -> int: ...
    def processEvents(
        self, hSSFRequest: HSSFRequest, inputStream: java.io.InputStream
    ) -> None: ...
    def processWorkbookEvents(
        self,
        hSSFRequest: HSSFRequest,
        pOIFSFileSystem: loci.poi.poifs.filesystem.POIFSFileSystem,
    ) -> None: ...

class HSSFListener:
    def processRecord(self, record: loci.poi.hssf.record.Record) -> None: ...

class HSSFRequest:
    def __init__(self): ...
    def addListener(
        self, hSSFListener: HSSFListener | typing.Callable, short: int
    ) -> None: ...
    def addListenerForAllRecords(
        self, hSSFListener: HSSFListener | typing.Callable
    ) -> None: ...

class HSSFUserException(java.lang.Exception):
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
    def getReason(self) -> java.lang.Throwable: ...

class AbortableHSSFListener(HSSFListener):
    def __init__(self): ...
    def abortableProcessRecord(self, record: loci.poi.hssf.record.Record) -> int: ...
    def processRecord(self, record: loci.poi.hssf.record.Record) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.hssf.eventusermodel")``.

    AbortableHSSFListener: type[AbortableHSSFListener]
    HSSFEventFactory: type[HSSFEventFactory]
    HSSFListener: type[HSSFListener]
    HSSFRequest: type[HSSFRequest]
    HSSFUserException: type[HSSFUserException]
