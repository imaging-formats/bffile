import typing
from typing import Protocol

import java.lang
import loci.formats
import loci.formats.ome
import ome.xml.meta
import ome.xml.model
import ome.xml.model.primitives

class BaseMetadata(ome.xml.meta.BaseMetadata): ...

class IMinMaxStore:
    def setChannelGlobalMinMax(
        self, int: int, double: float, double2: float, int2: int
    ) -> None: ...

class IPyramidStore:
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

class MetadataConverter:
    @staticmethod
    def convertMetadata(
        metadataRetrieve: ome.xml.meta.MetadataRetrieve,
        metadataStore: ome.xml.meta.MetadataStore,
    ) -> None: ...

class MetadataRetrieve(ome.xml.meta.MetadataRetrieve): ...
class MetadataStore(ome.xml.meta.MetadataStore): ...

class ModuloAnnotation(ome.xml.model.XMLAnnotation):
    MODULO_NS: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def setModulo(
        self,
        oMEXMLMetadata: loci.formats.ome.OMEXMLMetadata,
        modulo: loci.formats.Modulo,
    ) -> None: ...

class OriginalMetadataAnnotation(ome.xml.model.XMLAnnotation):
    ORIGINAL_METADATA_NS: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def getKey(self) -> java.lang.String: ...
    def getValueForKey(self) -> java.lang.String: ...
    def setKeyValue(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ) -> None: ...

class FilterMetadata(ome.xml.meta.FilterMetadata, MetadataStore):
    def __init__(self, metadataStore: MetadataStore, boolean: bool): ...

class IMetadata(ome.xml.meta.IMetadata, MetadataStore, MetadataRetrieve): ...

class DummyMetadata(ome.xml.meta.DummyMetadata, IMetadata):
    def __init__(self): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.formats.meta")``.

    BaseMetadata: type[BaseMetadata]
    DummyMetadata: type[DummyMetadata]
    FilterMetadata: type[FilterMetadata]
    IMetadata: type[IMetadata]
    IMinMaxStore: type[IMinMaxStore]
    IPyramidStore: type[IPyramidStore]
    MetadataConverter: type[MetadataConverter]
    MetadataRetrieve: type[MetadataRetrieve]
    MetadataStore: type[MetadataStore]
    ModuloAnnotation: type[ModuloAnnotation]
    OriginalMetadataAnnotation: type[OriginalMetadataAnnotation]
