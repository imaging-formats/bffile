import typing
from typing import Protocol

import java.lang
import loci.formats.meta
import ome.xml.meta
import ome.xml.model.primitives

class OMEXMLMetadata(loci.formats.meta.IMetadata, ome.xml.meta.OMEXMLMetadata): ...

class AbstractOMEXMLMetadata(ome.xml.meta.AbstractOMEXMLMetadata, OMEXMLMetadata):
    def __init__(self): ...

class OMEXMLMetadataImpl(ome.xml.meta.OMEXMLMetadataImpl, OMEXMLMetadata):
    def __init__(self): ...

class OMEPyramidStore(OMEXMLMetadataImpl, loci.formats.meta.IPyramidStore):
    NAMESPACE: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def dumpXML(self) -> java.lang.String: ...
    def getResolutionCount(self, int: int) -> int: ...
    def getResolutionSizeX(
        self, int: int, int2: int
    ) -> ome.xml.model.primitives.PositiveInteger: ...
    def getResolutionSizeY(
        self, int: int, int2: int
    ) -> ome.xml.model.primitives.PositiveInteger: ...
    def setResolutionSizeX(
        self,
        positiveInteger: ome.xml.model.primitives.PositiveInteger,
        int: int,
        int2: int,
    ) -> None: ...
    def setResolutionSizeY(
        self,
        positiveInteger: ome.xml.model.primitives.PositiveInteger,
        int: int,
        int2: int,
    ) -> None: ...
    def setRoot(self, metadataRoot: ome.xml.meta.MetadataRoot) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.formats.ome")``.

    AbstractOMEXMLMetadata: type[AbstractOMEXMLMetadata]
    OMEPyramidStore: type[OMEPyramidStore]
    OMEXMLMetadata: type[OMEXMLMetadata]
    OMEXMLMetadataImpl: type[OMEXMLMetadataImpl]
