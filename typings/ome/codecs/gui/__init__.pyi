import typing
from typing import Protocol

import java.awt.color
import java.awt.image
import java.lang
import jpype

class AWTImageTools:
    @typing.overload
    @staticmethod
    def constructImage(
        int: int,
        int2: int,
        int3: int,
        int4: int,
        boolean: bool,
        boolean2: bool,
        dataBuffer: java.awt.image.DataBuffer,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def constructImage(
        int: int,
        int2: int,
        int3: int,
        int4: int,
        boolean: bool,
        boolean2: bool,
        dataBuffer: java.awt.image.DataBuffer,
        colorModel: java.awt.image.ColorModel,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def getBytes(
        bufferedImage: java.awt.image.BufferedImage, boolean: bool
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def getBytes(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getBytes(
        writableRaster: java.awt.image.WritableRaster,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getBytes(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getDoubles(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getDoubles(
        writableRaster: java.awt.image.WritableRaster,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getDoubles(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getFloats(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getFloats(
        writableRaster: java.awt.image.WritableRaster,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getFloats(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    @staticmethod
    def getInts(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getInts(
        writableRaster: java.awt.image.WritableRaster,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getInts(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getPixelBytes(
        bufferedImage: java.awt.image.BufferedImage, boolean: bool
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getPixelBytes(
        bufferedImage: java.awt.image.BufferedImage,
        boolean: bool,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getPixelBytes(
        writableRaster: java.awt.image.WritableRaster, boolean: bool
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getPixelBytes(
        writableRaster: java.awt.image.WritableRaster,
        boolean: bool,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getPixels(bufferedImage: java.awt.image.BufferedImage) -> typing.Any: ...
    @typing.overload
    @staticmethod
    def getPixels(
        bufferedImage: java.awt.image.BufferedImage,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.Any: ...
    @typing.overload
    @staticmethod
    def getPixels(writableRaster: java.awt.image.WritableRaster) -> typing.Any: ...
    @typing.overload
    @staticmethod
    def getPixels(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.Any: ...
    @typing.overload
    @staticmethod
    def getShorts(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getShorts(
        writableRaster: java.awt.image.WritableRaster,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    @staticmethod
    def getShorts(
        writableRaster: java.awt.image.WritableRaster,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @staticmethod
    def makeColorModel(int: int, int2: int) -> java.awt.image.ColorModel: ...
    @staticmethod
    def makeColorSpace(int: int) -> java.awt.color.ColorSpace: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int, boolean: bool
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
        boolean2: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
        int4: int,
        boolean2: bool,
        boolean3: bool,
        boolean4: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int: int,
        int2: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
        boolean2: bool,
        boolean3: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        doubleArray: list[float] | jpype.JArray, int: int, int2: int
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        doubleArray: list[float] | jpype.JArray,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        doubleArray: list[typing.MutableSequence[float]] | jpype.JArray,
        int: int,
        int2: int,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        floatArray: list[float] | jpype.JArray, int: int, int2: int
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        floatArray: list[float] | jpype.JArray,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        floatArray: list[typing.MutableSequence[float]] | jpype.JArray,
        int: int,
        int2: int,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        intArray: list[int] | jpype.JArray, int2: int, int3: int, boolean: bool
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        intArray: list[int] | jpype.JArray,
        int2: int,
        int3: int,
        int4: int,
        boolean: bool,
        boolean2: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        intArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int2: int,
        int3: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        shortArray: list[int] | jpype.JArray, int: int, int2: int, boolean: bool
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        shortArray: list[int] | jpype.JArray,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
        boolean2: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        shortArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int: int,
        int2: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeRGBImage(
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
        int3: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeRGBImage(
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray, int: int, int2: int
    ) -> java.awt.image.BufferedImage: ...

class SignedByteBuffer(java.awt.image.DataBuffer):
    @typing.overload
    def __init__(self, byteArray: list[int] | jpype.JArray | bytes, int: int): ...
    @typing.overload
    def __init__(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray, int: int
    ): ...
    @typing.overload
    def getData(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getData(self, int: int) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getElem(self, int: int) -> int: ...
    @typing.overload
    def getElem(self, int: int, int2: int) -> int: ...
    @typing.overload
    def setElem(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setElem(self, int: int, int2: int, int3: int) -> None: ...

class SignedShortBuffer(java.awt.image.DataBuffer):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    @typing.overload
    def __init__(self, shortArray: list[int] | jpype.JArray, int: int): ...
    @typing.overload
    def __init__(self, shortArray: list[int] | jpype.JArray, int: int, int2: int): ...
    @typing.overload
    def __init__(
        self, shortArray: list[typing.MutableSequence[int]] | jpype.JArray, int: int
    ): ...
    @typing.overload
    def __init__(
        self,
        shortArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def getData(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getData(self, int: int) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getElem(self, int: int) -> int: ...
    @typing.overload
    def getElem(self, int: int, int2: int) -> int: ...
    @typing.overload
    def setElem(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setElem(self, int: int, int2: int, int3: int) -> None: ...

class TwoChannelColorSpace(java.awt.color.ColorSpace):
    CS_2C: typing.ClassVar[int] = ...
    def fromCIEXYZ(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    def fromRGB(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    @staticmethod
    def getInstance(int: int) -> java.awt.color.ColorSpace: ...
    def getName(self, int: int) -> java.lang.String: ...
    def getNumComponents(self) -> int: ...
    def getType(self) -> int: ...
    def isCS_sRGB(self) -> bool: ...
    def toCIEXYZ(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    def toRGB(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...

class UnsignedIntBuffer(java.awt.image.DataBuffer):
    @typing.overload
    def __init__(self, intArray: list[int] | jpype.JArray, int2: int): ...
    @typing.overload
    def __init__(
        self, intArray: list[typing.MutableSequence[int]] | jpype.JArray, int2: int
    ): ...
    @typing.overload
    def getData(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getData(self, int: int) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getElem(self, int: int) -> int: ...
    @typing.overload
    def getElem(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getElemDouble(self, int: int) -> float: ...
    @typing.overload
    def getElemDouble(self, int: int, int2: int) -> float: ...
    @typing.overload
    def getElemFloat(self, int: int) -> float: ...
    @typing.overload
    def getElemFloat(self, int: int, int2: int) -> float: ...
    @typing.overload
    def setElem(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setElem(self, int: int, int2: int, int3: int) -> None: ...
    @typing.overload
    def setElemDouble(self, int: int, double: float) -> None: ...
    @typing.overload
    def setElemDouble(self, int: int, int2: int, double: float) -> None: ...
    @typing.overload
    def setElemFloat(self, int: int, float: float) -> None: ...
    @typing.overload
    def setElemFloat(self, int: int, int2: int, float: float) -> None: ...

class UnsignedIntColorModel(java.awt.image.ColorModel):
    def __init__(self, int: int, int2: int, int3: int): ...
    def createCompatibleWritableRaster(
        self, int: int, int2: int
    ) -> java.awt.image.WritableRaster: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    @typing.overload
    def getDataElements(
        self, floatArray: list[float] | jpype.JArray, int: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, intArray: list[int] | jpype.JArray, int2: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(self, int: int, object: typing.Any) -> typing.Any: ...
    @typing.overload
    def getGreen(self, int: int) -> int: ...
    @typing.overload
    def getGreen(self, object: typing.Any) -> int: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    def isCompatibleRaster(self, raster: java.awt.image.Raster) -> bool: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.codecs.gui")``.

    AWTImageTools: type[AWTImageTools]
    SignedByteBuffer: type[SignedByteBuffer]
    SignedShortBuffer: type[SignedShortBuffer]
    TwoChannelColorSpace: type[TwoChannelColorSpace]
    UnsignedIntBuffer: type[UnsignedIntBuffer]
    UnsignedIntColorModel: type[UnsignedIntColorModel]
