import typing
from typing import Protocol

import java.io
import jpype
import loci.poi.hssf.model
import loci.poi.hssf.record

class ERFListener:
    def processRecord(self, record: loci.poi.hssf.record.Record) -> bool: ...

class EventRecordFactory:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, boolean: bool): ...
    @staticmethod
    def createRecord(
        recordInputStream: loci.poi.hssf.record.RecordInputStream,
    ) -> typing.MutableSequence[loci.poi.hssf.record.Record]: ...
    @staticmethod
    def getAllKnownRecordSIDs() -> typing.MutableSequence[int]: ...
    def processRecords(self, inputStream: java.io.InputStream) -> None: ...
    def registerListener(
        self,
        eRFListener: ERFListener | typing.Callable,
        shortArray: list[int] | jpype.JArray,
    ) -> None: ...

class ModelFactoryListener:
    def process(self, model: loci.poi.hssf.model.Model) -> bool: ...

class ModelFactory(ERFListener):
    def __init__(self): ...
    def processRecord(self, record: loci.poi.hssf.record.Record) -> bool: ...
    def registerListener(
        self, modelFactoryListener: ModelFactoryListener | typing.Callable
    ) -> None: ...
    def run(self, inputStream: java.io.InputStream) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.hssf.eventmodel")``.

    ERFListener: type[ERFListener]
    EventRecordFactory: type[EventRecordFactory]
    ModelFactory: type[ModelFactory]
    ModelFactoryListener: type[ModelFactoryListener]
