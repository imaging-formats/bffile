import typing
from typing import Protocol

import java.lang
import jpype
import loci.common
import loci.formats
import loci.formats.codec
import loci.formats.tiff

class APNGWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class AVIWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class CellH5Writer(loci.formats.FormatWriter):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        stringArray: list[java.lang.String] | jpype.JArray,
    ): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    def getPlaneCount(self) -> int: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class EPSWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...

class ICSWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...
    def setOutputOrder(self, string: java.lang.String | str) -> None: ...

class IExtraMetadataWriter:
    def setExtraMetadata(self, string: java.lang.String | str) -> None: ...

class ImageIOWriter(loci.formats.FormatWriter):
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        stringArray: list[java.lang.String] | jpype.JArray,
        string3: java.lang.String | str,
    ): ...
    def getNativeDataType(self) -> type[typing.Any]: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def savePlane(self, int: int, object: typing.Any) -> None: ...
    @typing.overload
    def savePlane(
        self, int: int, object: typing.Any, region: loci.common.Region
    ) -> None: ...
    @typing.overload
    def savePlane(
        self, int: int, object: typing.Any, int2: int, int3: int, int4: int, int5: int
    ) -> None: ...

class JPEG2000Writer(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def compressBuffer(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...

class JavaWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class OMEXMLWriter(loci.formats.FormatWriter):
    CREATOR_KEY: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    def preserveCreator(self) -> bool: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class QTWriter(loci.formats.FormatWriter):
    CODEC_MOTION_JPEG_B: typing.ClassVar[int] = ...
    CODEC_CINEPAK: typing.ClassVar[int] = ...
    CODEC_ANIMATION: typing.ClassVar[int] = ...
    CODEC_H_263: typing.ClassVar[int] = ...
    CODEC_SORENSON: typing.ClassVar[int] = ...
    CODEC_SORENSON_3: typing.ClassVar[int] = ...
    CODEC_MPEG_4: typing.ClassVar[int] = ...
    CODEC_RAW: typing.ClassVar[int] = ...
    QUALITY_LOW: typing.ClassVar[int] = ...
    QUALITY_NORMAL: typing.ClassVar[int] = ...
    QUALITY_HIGH: typing.ClassVar[int] = ...
    QUALITY_MAXIMUM: typing.ClassVar[int] = ...
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setCodec(self, int: int) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...
    def setQuality(self, int: int) -> None: ...

class TiffWriter(loci.formats.FormatWriter):
    COMPRESSION_UNCOMPRESSED: typing.ClassVar[java.lang.String] = ...
    COMPRESSION_LZW: typing.ClassVar[java.lang.String] = ...
    COMPRESSION_J2K: typing.ClassVar[java.lang.String] = ...
    COMPRESSION_J2K_LOSSY: typing.ClassVar[java.lang.String] = ...
    COMPRESSION_JPEG: typing.ClassVar[java.lang.String] = ...
    COMPRESSION_ZLIB: typing.ClassVar[java.lang.String] = ...
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        stringArray: list[java.lang.String] | jpype.JArray,
    ): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    def getCodec(self) -> loci.formats.codec.Codec: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    def getPlaneCount(self) -> int: ...
    def getTileSizeX(self) -> int: ...
    def getTileSizeY(self) -> int: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def saveCompressedBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setBigTiff(self, boolean: bool) -> None: ...
    def setCanDetectBigTiff(self, boolean: bool) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...
    def setTileSizeX(self, int: int) -> int: ...
    def setTileSizeY(self, int: int) -> int: ...

class V3DrawWriter(loci.formats.FormatWriter):
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...
    def setOutputOrder(self, string: java.lang.String | str) -> None: ...

class DicomWriter(loci.formats.FormatWriter, IExtraMetadataWriter):
    UID_ROOT_KEY: typing.ClassVar[java.lang.String] = ...
    UID_DEFAULT_ROOT: typing.ClassVar[java.lang.String] = ...
    TIFF_KEY: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def canDoStacks(self) -> bool: ...
    def close(self) -> None: ...
    def getCodec(self) -> loci.formats.codec.Codec: ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    def getTileSizeX(self) -> int: ...
    def getTileSizeY(self) -> int: ...
    def getUIDRoot(self) -> java.lang.String: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def saveCompressedBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setBigTiff(self, boolean: bool) -> None: ...
    def setExtraMetadata(self, string: java.lang.String | str) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...
    def setResolution(self, int: int) -> None: ...
    def setSeries(self, int: int) -> None: ...
    def setTileSizeX(self, int: int) -> int: ...
    def setTileSizeY(self, int: int) -> int: ...
    def writeDualPersonality(self) -> bool: ...

class JPEGWriter(ImageIOWriter):
    def __init__(self): ...
    @typing.overload
    def getPixelTypes(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixelTypes(
        self, string: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...

class OMETiffWriter(TiffWriter):
    COMPANION_KEY: typing.ClassVar[java.lang.String] = ...
    CREATOR_KEY: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    def close(self) -> None: ...
    def getCompanion(self) -> java.lang.String: ...
    def preserveCreator(self) -> bool: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
    ) -> None: ...
    def saveCompressedBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    def setId(self, string: java.lang.String | str) -> None: ...

class PyramidOMETiffWriter(OMETiffWriter):
    def __init__(self): ...
    def close(self) -> None: ...
    def isThisType(self, string: java.lang.String | str) -> bool: ...
    @typing.overload
    def saveBytes(
        self, int: int, byteArray: list[int] | jpype.JArray | bytes
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        region: loci.common.Region,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @typing.overload
    def saveBytes(
        self,
        int: int,
        byteArray: list[int] | jpype.JArray | bytes,
        iFD: loci.formats.tiff.IFD,
    ) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.formats.out")``.

    APNGWriter: type[APNGWriter]
    AVIWriter: type[AVIWriter]
    CellH5Writer: type[CellH5Writer]
    DicomWriter: type[DicomWriter]
    EPSWriter: type[EPSWriter]
    ICSWriter: type[ICSWriter]
    IExtraMetadataWriter: type[IExtraMetadataWriter]
    ImageIOWriter: type[ImageIOWriter]
    JPEG2000Writer: type[JPEG2000Writer]
    JPEGWriter: type[JPEGWriter]
    JavaWriter: type[JavaWriter]
    OMETiffWriter: type[OMETiffWriter]
    OMEXMLWriter: type[OMEXMLWriter]
    PyramidOMETiffWriter: type[PyramidOMETiffWriter]
    QTWriter: type[QTWriter]
    TiffWriter: type[TiffWriter]
    V3DrawWriter: type[V3DrawWriter]
