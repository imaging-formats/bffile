import typing
from typing import Protocol

import java.awt.image
import java.lang
import java.nio
import jpype
import loci.common
import loci.common.enumeration
import ome.codecs.gui
import ome.codecs.services

class BitBuffer:
    def __init__(self, byteArray: list[int] | jpype.JArray | bytes): ...
    def getBits(self, int: int) -> int: ...
    def getByteBuffer(self) -> typing.MutableSequence[int]: ...
    def isBitOnByteBoundary(self) -> bool: ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def skipBits(self, long: int) -> None: ...

class BitWriter:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def toByteArray(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def write(self, int: int, int2: int) -> None: ...
    @typing.overload
    def write(self, string: java.lang.String | str) -> None: ...

class ByteVector:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, byteArray: list[int] | jpype.JArray | bytes): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def add(self, byte: int) -> None: ...
    @typing.overload
    def add(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    @typing.overload
    def add(
        self, byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> None: ...
    def clear(self) -> None: ...
    def get(self, int: int) -> int: ...
    def size(self) -> int: ...
    def toByteArray(self) -> typing.MutableSequence[int]: ...

class Codec:
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class CodecException(java.lang.Exception):
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

class CodecOptions:
    width: int = ...
    height: int = ...
    channels: int = ...
    bitsPerSample: int = ...
    littleEndian: bool = ...
    interleaved: bool = ...
    signed: bool = ...
    tileWidth: int = ...
    tileHeight: int = ...
    tileGridXOffset: int = ...
    tileGridYOffset: int = ...
    maxBytes: int = ...
    previousImage: typing.MutableSequence[int] = ...
    lossless: bool = ...
    colorModel: java.awt.image.ColorModel = ...
    quality: float = ...
    ycbcr: bool = ...
    disableChromaSubsampling: bool = ...
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, codecOptions: CodecOptions): ...
    @staticmethod
    def getDefaultOptions() -> CodecOptions: ...

class CompressionType(
    java.lang.Enum["CompressionType"], loci.common.enumeration.CodedEnum
):
    UNCOMPRESSED: typing.ClassVar[CompressionType] = ...
    ZLIB: typing.ClassVar[CompressionType] = ...
    CINEPAK: typing.ClassVar[CompressionType] = ...
    ANIMATION: typing.ClassVar[CompressionType] = ...
    H_263: typing.ClassVar[CompressionType] = ...
    SORENSON: typing.ClassVar[CompressionType] = ...
    SORENSON_3: typing.ClassVar[CompressionType] = ...
    MPEG_4: typing.ClassVar[CompressionType] = ...
    LZW: typing.ClassVar[CompressionType] = ...
    J2K: typing.ClassVar[CompressionType] = ...
    J2K_LOSSY: typing.ClassVar[CompressionType] = ...
    JPEG: typing.ClassVar[CompressionType] = ...
    @staticmethod
    def get(int: int) -> CompressionType: ...
    def getCode(self) -> int: ...
    def getCompression(self) -> java.lang.String: ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> CompressionType: ...
    @staticmethod
    def values() -> typing.MutableSequence[CompressionType]: ...

class JPEG2000BoxType(
    java.lang.Enum["JPEG2000BoxType"], loci.common.enumeration.CodedEnum
):
    SIGNATURE: typing.ClassVar[JPEG2000BoxType] = ...
    SIGNATURE_WRONG_ENDIANNESS: typing.ClassVar[JPEG2000BoxType] = ...
    FILE: typing.ClassVar[JPEG2000BoxType] = ...
    HEADER: typing.ClassVar[JPEG2000BoxType] = ...
    IMAGE_HEADER: typing.ClassVar[JPEG2000BoxType] = ...
    BITS_PER_COMPONENT: typing.ClassVar[JPEG2000BoxType] = ...
    COLOUR_SPECIFICATION: typing.ClassVar[JPEG2000BoxType] = ...
    PALETTE: typing.ClassVar[JPEG2000BoxType] = ...
    COMPONENT_MAPPING: typing.ClassVar[JPEG2000BoxType] = ...
    CHANNEL_DEFINITION: typing.ClassVar[JPEG2000BoxType] = ...
    RESOLUTION: typing.ClassVar[JPEG2000BoxType] = ...
    CAPTURE_RESOLUTION: typing.ClassVar[JPEG2000BoxType] = ...
    DEFAULT_DISPLAY_RESOLUTION: typing.ClassVar[JPEG2000BoxType] = ...
    CONTIGUOUS_CODESTREAM: typing.ClassVar[JPEG2000BoxType] = ...
    INTELLECTUAL_PROPERTY: typing.ClassVar[JPEG2000BoxType] = ...
    XML: typing.ClassVar[JPEG2000BoxType] = ...
    UUID: typing.ClassVar[JPEG2000BoxType] = ...
    UUID_INFO: typing.ClassVar[JPEG2000BoxType] = ...
    UUID_LIST: typing.ClassVar[JPEG2000BoxType] = ...
    URL: typing.ClassVar[JPEG2000BoxType] = ...
    ASSOCIATION: typing.ClassVar[JPEG2000BoxType] = ...
    LABEL: typing.ClassVar[JPEG2000BoxType] = ...
    PLACEHOLDER: typing.ClassVar[JPEG2000BoxType] = ...
    @staticmethod
    def get(int: int) -> JPEG2000BoxType: ...
    def getCode(self) -> int: ...
    def getName(self) -> java.lang.String: ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> JPEG2000BoxType: ...
    @staticmethod
    def values() -> typing.MutableSequence[JPEG2000BoxType]: ...

class JPEG2000SegmentMarker(
    java.lang.Enum["JPEG2000SegmentMarker"], loci.common.enumeration.CodedEnum
):
    RESERVED_DELIMITER_MARKER_MIN: typing.ClassVar[JPEG2000SegmentMarker] = ...
    RESERVED_DELIMITER_MARKER_MAX: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SOC: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SOC_WRONG_ENDIANNESS: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SOT: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SOD: typing.ClassVar[JPEG2000SegmentMarker] = ...
    EOC: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SIZ: typing.ClassVar[JPEG2000SegmentMarker] = ...
    COD: typing.ClassVar[JPEG2000SegmentMarker] = ...
    COC: typing.ClassVar[JPEG2000SegmentMarker] = ...
    RGN: typing.ClassVar[JPEG2000SegmentMarker] = ...
    QCD: typing.ClassVar[JPEG2000SegmentMarker] = ...
    QCC: typing.ClassVar[JPEG2000SegmentMarker] = ...
    POC: typing.ClassVar[JPEG2000SegmentMarker] = ...
    TLM: typing.ClassVar[JPEG2000SegmentMarker] = ...
    PLM: typing.ClassVar[JPEG2000SegmentMarker] = ...
    PLT: typing.ClassVar[JPEG2000SegmentMarker] = ...
    PPM: typing.ClassVar[JPEG2000SegmentMarker] = ...
    PPT: typing.ClassVar[JPEG2000SegmentMarker] = ...
    SOP: typing.ClassVar[JPEG2000SegmentMarker] = ...
    EPH: typing.ClassVar[JPEG2000SegmentMarker] = ...
    CRG: typing.ClassVar[JPEG2000SegmentMarker] = ...
    COM: typing.ClassVar[JPEG2000SegmentMarker] = ...
    @staticmethod
    def get(int: int) -> JPEG2000SegmentMarker: ...
    def getCode(self) -> int: ...
    def getName(self) -> java.lang.String: ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> JPEG2000SegmentMarker: ...
    @staticmethod
    def values() -> typing.MutableSequence[JPEG2000SegmentMarker]: ...

class JPEGTileDecoder(java.lang.AutoCloseable):
    def __init__(self): ...
    def close(self) -> None: ...
    def getHeight(self) -> int: ...
    def getScanline(self, int: int) -> typing.MutableSequence[int]: ...
    def getWidth(self) -> int: ...
    @typing.overload
    def initialize(self, string: java.lang.String | str, int: int) -> None: ...
    @typing.overload
    def initialize(
        self, randomAccessInputStream: loci.common.RandomAccessInputStream, int: int
    ) -> None: ...
    @typing.overload
    def initialize(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        int: int,
        int2: int,
        int3: int,
    ) -> None: ...
    def preprocess(
        self, randomAccessInputStream: loci.common.RandomAccessInputStream
    ) -> typing.MutableSequence[int]: ...

class BaseCodec(Codec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    def test(self) -> None: ...

class HuffmanCodecOptions(CodecOptions):
    table: typing.MutableSequence[int] = ...
    def __init__(self): ...

class JPEG2000CodecOptions(CodecOptions):
    codeBlockSize: typing.MutableSequence[int] = ...
    numDecompositionLevels: int = ...
    resolution: int = ...
    writeBox: bool = ...
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, codecOptions: CodecOptions): ...
    @typing.overload
    @staticmethod
    def getDefaultOptions() -> CodecOptions: ...
    @typing.overload
    @staticmethod
    def getDefaultOptions() -> JPEG2000CodecOptions: ...
    @typing.overload
    @staticmethod
    def getDefaultOptions(codecOptions: CodecOptions) -> JPEG2000CodecOptions: ...

class MJPBCodecOptions(CodecOptions):
    interlaced: bool = ...
    def __init__(self): ...

class MissingLibraryException(CodecException):
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

class UnsupportedCompressionException(CodecException):
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

class Base64Codec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class HuffmanCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getSample(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> int: ...
    @typing.overload
    def getSample(self, bitBuffer: BitBuffer, codecOptions: CodecOptions) -> int: ...

class JPEG2000Codec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class JPEGCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class LZ4Codec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        int3: int,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class LZOCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class LZWCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class LosslessJPEGCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class MJPBCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class MSRLECodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class MSVideoCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class PackbitsCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class PassthroughCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class QTRLECodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class RPZACodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class ZlibCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class ZstdCodec(BaseCodec):
    def __init__(self): ...
    @typing.overload
    def compress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def compress(
        self,
        byteBuffer: java.nio.ByteBuffer,
        byteArray: list[int] | jpype.JArray | bytes,
        codecOptions: CodecOptions,
    ) -> java.nio.ByteBuffer: ...
    @typing.overload
    def decompress(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self, byteArray: list[int] | jpype.JArray | bytes, codecOptions: CodecOptions
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def decompress(
        self,
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        codecOptions: CodecOptions,
    ) -> typing.MutableSequence[int]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.codecs")``.

    Base64Codec: type[Base64Codec]
    BaseCodec: type[BaseCodec]
    BitBuffer: type[BitBuffer]
    BitWriter: type[BitWriter]
    ByteVector: type[ByteVector]
    Codec: type[Codec]
    CodecException: type[CodecException]
    CodecOptions: type[CodecOptions]
    CompressionType: type[CompressionType]
    HuffmanCodec: type[HuffmanCodec]
    HuffmanCodecOptions: type[HuffmanCodecOptions]
    JPEG2000BoxType: type[JPEG2000BoxType]
    JPEG2000Codec: type[JPEG2000Codec]
    JPEG2000CodecOptions: type[JPEG2000CodecOptions]
    JPEG2000SegmentMarker: type[JPEG2000SegmentMarker]
    JPEGCodec: type[JPEGCodec]
    JPEGTileDecoder: type[JPEGTileDecoder]
    LZ4Codec: type[LZ4Codec]
    LZOCodec: type[LZOCodec]
    LZWCodec: type[LZWCodec]
    LosslessJPEGCodec: type[LosslessJPEGCodec]
    MJPBCodec: type[MJPBCodec]
    MJPBCodecOptions: type[MJPBCodecOptions]
    MSRLECodec: type[MSRLECodec]
    MSVideoCodec: type[MSVideoCodec]
    MissingLibraryException: type[MissingLibraryException]
    PackbitsCodec: type[PackbitsCodec]
    PassthroughCodec: type[PassthroughCodec]
    QTRLECodec: type[QTRLECodec]
    RPZACodec: type[RPZACodec]
    UnsupportedCompressionException: type[UnsupportedCompressionException]
    ZlibCodec: type[ZlibCodec]
    ZstdCodec: type[ZstdCodec]
    gui: ome.codecs.gui.__module_protocol__
    services: ome.codecs.services.__module_protocol__
