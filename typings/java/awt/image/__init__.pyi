import typing
from typing import Protocol

import java.awt
import java.awt.color
import java.awt.geom
import java.awt.image.renderable
import java.lang
import java.math
import java.util
import jpype

class BufferStrategy:
    def contentsLost(self) -> bool: ...
    def contentsRestored(self) -> bool: ...
    def dispose(self) -> None: ...
    def getCapabilities(self) -> java.awt.BufferCapabilities: ...
    def getDrawGraphics(self) -> java.awt.Graphics: ...
    def show(self) -> None: ...

class BufferedImageOp:
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...

class ColorModel(java.awt.Transparency):
    def __init__(self, int: int): ...
    def coerceData(
        self, writableRaster: WritableRaster, boolean: bool
    ) -> ColorModel: ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createCompatibleWritableRaster(self, int: int, int2: int) -> WritableRaster: ...
    def equals(self, object: typing.Any) -> bool: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    def getAlphaRaster(self, writableRaster: WritableRaster) -> WritableRaster: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    def getColorSpace(self) -> java.awt.color.ColorSpace: ...
    @typing.overload
    def getComponentSize(self, int: int) -> int: ...
    @typing.overload
    def getComponentSize(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, int: int, intArray: list[int] | jpype.JArray, int3: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, object: typing.Any, intArray: list[int] | jpype.JArray, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElement(
        self, floatArray: list[float] | jpype.JArray, int: int
    ) -> int: ...
    @typing.overload
    def getDataElement(self, intArray: list[int] | jpype.JArray, int2: int) -> int: ...
    @typing.overload
    def getDataElements(
        self, floatArray: list[float] | jpype.JArray, int: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(self, int: int, object: typing.Any) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, intArray: list[int] | jpype.JArray, int2: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getGreen(self, int: int) -> int: ...
    @typing.overload
    def getGreen(self, object: typing.Any) -> int: ...
    @typing.overload
    def getNormalizedComponents(
        self,
        intArray: list[int] | jpype.JArray,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        int3: int,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getNormalizedComponents(
        self, object: typing.Any, floatArray: list[float] | jpype.JArray, int: int
    ) -> typing.MutableSequence[float]: ...
    def getNumColorComponents(self) -> int: ...
    def getNumComponents(self) -> int: ...
    def getPixelSize(self) -> int: ...
    @typing.overload
    def getRGB(self, int: int) -> int: ...
    @typing.overload
    def getRGB(self, object: typing.Any) -> int: ...
    @staticmethod
    def getRGBdefault() -> ColorModel: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    def getTransferType(self) -> int: ...
    def getTransparency(self) -> int: ...
    def getUnnormalizedComponents(
        self,
        floatArray: list[float] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
        int3: int,
    ) -> typing.MutableSequence[int]: ...
    def hasAlpha(self) -> bool: ...
    def hashCode(self) -> int: ...
    def isAlphaPremultiplied(self) -> bool: ...
    def isCompatibleRaster(self, raster: Raster) -> bool: ...
    def isCompatibleSampleModel(self, sampleModel: SampleModel) -> bool: ...
    def toString(self) -> java.lang.String: ...

class DataBuffer:
    TYPE_BYTE: typing.ClassVar[int] = ...
    TYPE_USHORT: typing.ClassVar[int] = ...
    TYPE_SHORT: typing.ClassVar[int] = ...
    TYPE_INT: typing.ClassVar[int] = ...
    TYPE_FLOAT: typing.ClassVar[int] = ...
    TYPE_DOUBLE: typing.ClassVar[int] = ...
    TYPE_UNDEFINED: typing.ClassVar[int] = ...
    def getDataType(self) -> int: ...
    @staticmethod
    def getDataTypeSize(int: int) -> int: ...
    @typing.overload
    def getElem(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getElem(self, int: int) -> int: ...
    @typing.overload
    def getElemDouble(self, int: int) -> float: ...
    @typing.overload
    def getElemDouble(self, int: int, int2: int) -> float: ...
    @typing.overload
    def getElemFloat(self, int: int) -> float: ...
    @typing.overload
    def getElemFloat(self, int: int, int2: int) -> float: ...
    def getNumBanks(self) -> int: ...
    def getOffset(self) -> int: ...
    def getOffsets(self) -> typing.MutableSequence[int]: ...
    def getSize(self) -> int: ...
    @typing.overload
    def setElem(self, int: int, int2: int, int3: int) -> None: ...
    @typing.overload
    def setElem(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setElemDouble(self, int: int, double: float) -> None: ...
    @typing.overload
    def setElemDouble(self, int: int, int2: int, double: float) -> None: ...
    @typing.overload
    def setElemFloat(self, int: int, float: float) -> None: ...
    @typing.overload
    def setElemFloat(self, int: int, int2: int, float: float) -> None: ...

class ImageConsumer:
    RANDOMPIXELORDER: typing.ClassVar[int] = ...
    TOPDOWNLEFTRIGHT: typing.ClassVar[int] = ...
    COMPLETESCANLINES: typing.ClassVar[int] = ...
    SINGLEPASS: typing.ClassVar[int] = ...
    SINGLEFRAME: typing.ClassVar[int] = ...
    IMAGEERROR: typing.ClassVar[int] = ...
    SINGLEFRAMEDONE: typing.ClassVar[int] = ...
    STATICIMAGEDONE: typing.ClassVar[int] = ...
    IMAGEABORTED: typing.ClassVar[int] = ...
    def imageComplete(self, int: int) -> None: ...
    def setColorModel(self, colorModel: ColorModel) -> None: ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    def setHints(self, int: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setProperties(
        self, hashtable: java.util.Hashtable[typing.Any, typing.Any]
    ) -> None: ...

class ImageObserver:
    WIDTH: typing.ClassVar[int] = ...
    HEIGHT: typing.ClassVar[int] = ...
    PROPERTIES: typing.ClassVar[int] = ...
    SOMEBITS: typing.ClassVar[int] = ...
    FRAMEBITS: typing.ClassVar[int] = ...
    ALLBITS: typing.ClassVar[int] = ...
    ERROR: typing.ClassVar[int] = ...
    ABORT: typing.ClassVar[int] = ...
    def imageUpdate(
        self,
        image: java.awt.Image,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> bool: ...

class ImageProducer:
    def addConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def isConsumer(self, imageConsumer: ImageConsumer) -> bool: ...
    def removeConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def requestTopDownLeftRightResend(self, imageConsumer: ImageConsumer) -> None: ...
    def startProduction(self, imageConsumer: ImageConsumer) -> None: ...

class ImagingOpException(java.lang.RuntimeException):
    def __init__(self, string: java.lang.String | str): ...

class Kernel(java.lang.Cloneable):
    def __init__(self, int: int, int2: int, floatArray: list[float] | jpype.JArray): ...
    def clone(self) -> typing.Any: ...
    def getHeight(self) -> int: ...
    def getKernelData(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    def getWidth(self) -> int: ...
    def getXOrigin(self) -> int: ...
    def getYOrigin(self) -> int: ...

class LookupTable:
    def getNumComponents(self) -> int: ...
    def getOffset(self) -> int: ...
    def lookupPixel(
        self, intArray: list[int] | jpype.JArray, intArray2: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...

class MultiResolutionImage:
    def getResolutionVariant(self, double: float, double2: float) -> java.awt.Image: ...
    def getResolutionVariants(self) -> java.util.List[java.awt.Image]: ...

class Raster:
    @typing.overload
    @staticmethod
    def createBandedRaster(
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        intArray2: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createBandedRaster(
        int: int, int2: int, int3: int, int4: int, point: java.awt.Point
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createBandedRaster(
        dataBuffer: DataBuffer,
        int: int,
        int2: int,
        int3: int,
        intArray: list[int] | jpype.JArray,
        intArray2: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    def createChild(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        int6: int,
        intArray: list[int] | jpype.JArray,
    ) -> Raster: ...
    @typing.overload
    def createCompatibleWritableRaster(self) -> WritableRaster: ...
    @typing.overload
    def createCompatibleWritableRaster(self, int: int, int2: int) -> WritableRaster: ...
    @typing.overload
    def createCompatibleWritableRaster(
        self, int: int, int2: int, int3: int, int4: int
    ) -> WritableRaster: ...
    @typing.overload
    def createCompatibleWritableRaster(
        self, rectangle: java.awt.Rectangle
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createInterleavedRaster(
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createInterleavedRaster(
        int: int, int2: int, int3: int, int4: int, point: java.awt.Point
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createInterleavedRaster(
        dataBuffer: DataBuffer,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createPackedRaster(
        int: int, int2: int, int3: int, int4: int, int5: int, point: java.awt.Point
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createPackedRaster(
        int: int,
        int2: int,
        int3: int,
        intArray: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createPackedRaster(
        dataBuffer: DataBuffer,
        int: int,
        int2: int,
        int3: int,
        intArray: list[int] | jpype.JArray,
        point: java.awt.Point,
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createPackedRaster(
        dataBuffer: DataBuffer, int: int, int2: int, int3: int, point: java.awt.Point
    ) -> WritableRaster: ...
    @staticmethod
    def createRaster(
        sampleModel: SampleModel, dataBuffer: DataBuffer, point: java.awt.Point
    ) -> Raster: ...
    def createTranslatedChild(self, int: int, int2: int) -> Raster: ...
    @typing.overload
    @staticmethod
    def createWritableRaster(
        sampleModel: SampleModel, point: java.awt.Point
    ) -> WritableRaster: ...
    @typing.overload
    @staticmethod
    def createWritableRaster(
        sampleModel: SampleModel, dataBuffer: DataBuffer, point: java.awt.Point
    ) -> WritableRaster: ...
    def getBounds(self) -> java.awt.Rectangle: ...
    def getDataBuffer(self) -> DataBuffer: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, int3: int, int4: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any
    ) -> typing.Any: ...
    def getHeight(self) -> int: ...
    def getMinX(self) -> int: ...
    def getMinY(self) -> int: ...
    def getNumBands(self) -> int: ...
    def getNumDataElements(self) -> int: ...
    def getParent(self) -> Raster: ...
    @typing.overload
    def getPixel(
        self, int: int, int2: int, doubleArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self, int: int, int2: int, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self, int: int, int2: int, intArray: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
    ) -> typing.MutableSequence[int]: ...
    def getSample(self, int: int, int2: int, int3: int) -> int: ...
    def getSampleDouble(self, int: int, int2: int, int3: int) -> float: ...
    def getSampleFloat(self, int: int, int2: int, int3: int) -> float: ...
    def getSampleModel(self) -> SampleModel: ...
    def getSampleModelTranslateX(self) -> int: ...
    def getSampleModelTranslateY(self) -> int: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
    ) -> typing.MutableSequence[int]: ...
    def getTransferType(self) -> int: ...
    def getWidth(self) -> int: ...

class RasterFormatException(java.lang.RuntimeException):
    def __init__(self, string: java.lang.String | str): ...

class RasterOp:
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...

class RenderedImage:
    def copyData(self, writableRaster: WritableRaster) -> WritableRaster: ...
    def getColorModel(self) -> ColorModel: ...
    @typing.overload
    def getData(self) -> Raster: ...
    @typing.overload
    def getData(self, rectangle: java.awt.Rectangle) -> Raster: ...
    def getHeight(self) -> int: ...
    def getMinTileX(self) -> int: ...
    def getMinTileY(self) -> int: ...
    def getMinX(self) -> int: ...
    def getMinY(self) -> int: ...
    def getNumXTiles(self) -> int: ...
    def getNumYTiles(self) -> int: ...
    def getProperty(self, string: java.lang.String | str) -> typing.Any: ...
    def getPropertyNames(self) -> typing.MutableSequence[java.lang.String]: ...
    def getSampleModel(self) -> SampleModel: ...
    def getSources(self) -> java.util.Vector[RenderedImage]: ...
    def getTile(self, int: int, int2: int) -> Raster: ...
    def getTileGridXOffset(self) -> int: ...
    def getTileGridYOffset(self) -> int: ...
    def getTileHeight(self) -> int: ...
    def getTileWidth(self) -> int: ...
    def getWidth(self) -> int: ...

class SampleModel:
    def __init__(self, int: int, int2: int, int3: int, int4: int): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createDataBuffer(self) -> DataBuffer: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> typing.Any: ...
    def getDataType(self) -> int: ...
    def getHeight(self) -> int: ...
    def getNumBands(self) -> int: ...
    def getNumDataElements(self) -> int: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getSample(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> int: ...
    def getSampleDouble(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    def getSampleFloat(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    @typing.overload
    def getSampleSize(self, int: int) -> int: ...
    @typing.overload
    def getSampleSize(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getTransferType(self) -> int: ...
    def getWidth(self) -> int: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, int4: int, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, double: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, float: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...

class TileObserver:
    def tileUpdate(
        self,
        writableRenderedImage: WritableRenderedImage,
        int: int,
        int2: int,
        boolean: bool,
    ) -> None: ...

class VolatileImage(java.awt.Image, java.awt.Transparency):
    IMAGE_OK: typing.ClassVar[int] = ...
    IMAGE_RESTORED: typing.ClassVar[int] = ...
    IMAGE_INCOMPATIBLE: typing.ClassVar[int] = ...
    def contentsLost(self) -> bool: ...
    def createGraphics(self) -> java.awt.Graphics2D: ...
    @typing.overload
    def getCapabilities(self) -> java.awt.ImageCapabilities: ...
    @typing.overload
    def getCapabilities(
        self, graphicsConfiguration: java.awt.GraphicsConfiguration
    ) -> java.awt.ImageCapabilities: ...
    def getGraphics(self) -> java.awt.Graphics: ...
    @typing.overload
    def getHeight(self, imageObserver: ImageObserver | typing.Callable) -> int: ...
    @typing.overload
    def getHeight(self) -> int: ...
    def getSnapshot(self) -> BufferedImage: ...
    def getSource(self) -> ImageProducer: ...
    def getTransparency(self) -> int: ...
    @typing.overload
    def getWidth(self, imageObserver: ImageObserver | typing.Callable) -> int: ...
    @typing.overload
    def getWidth(self) -> int: ...
    def validate(
        self, graphicsConfiguration: java.awt.GraphicsConfiguration
    ) -> int: ...

class AbstractMultiResolutionImage(java.awt.Image, MultiResolutionImage):
    def getGraphics(self) -> java.awt.Graphics: ...
    def getHeight(self, imageObserver: ImageObserver | typing.Callable) -> int: ...
    def getProperty(
        self,
        string: java.lang.String | str,
        imageObserver: ImageObserver | typing.Callable,
    ) -> typing.Any: ...
    def getSource(self) -> ImageProducer: ...
    def getWidth(self, imageObserver: ImageObserver | typing.Callable) -> int: ...

class AffineTransformOp(BufferedImageOp, RasterOp):
    TYPE_NEAREST_NEIGHBOR: typing.ClassVar[int] = ...
    TYPE_BILINEAR: typing.ClassVar[int] = ...
    TYPE_BICUBIC: typing.ClassVar[int] = ...
    @typing.overload
    def __init__(self, affineTransform: java.awt.geom.AffineTransform, int: int): ...
    @typing.overload
    def __init__(
        self,
        affineTransform: java.awt.geom.AffineTransform,
        renderingHints: java.awt.RenderingHints,
    ): ...
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    @typing.overload
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    @typing.overload
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    @typing.overload
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    @typing.overload
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getInterpolationType(self) -> int: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...
    def getTransform(self) -> java.awt.geom.AffineTransform: ...

class BandCombineOp(RasterOp):
    def __init__(
        self,
        floatArray: list[typing.MutableSequence[float]] | jpype.JArray,
        renderingHints: java.awt.RenderingHints,
    ): ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getMatrix(self) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...

class ByteLookupTable(LookupTable):
    @typing.overload
    def __init__(self, int: int, byteArray: list[int] | jpype.JArray | bytes): ...
    @typing.overload
    def __init__(
        self, int: int, byteArray: list[typing.MutableSequence[int]] | jpype.JArray
    ): ...
    def getTable(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    def lookupPixel(
        self,
        byteArray: list[int] | jpype.JArray | bytes,
        byteArray2: list[int] | jpype.JArray | bytes,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def lookupPixel(
        self, intArray: list[int] | jpype.JArray, intArray2: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...

class ColorConvertOp(BufferedImageOp, RasterOp):
    @typing.overload
    def __init__(self, renderingHints: java.awt.RenderingHints): ...
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        renderingHints: java.awt.RenderingHints,
    ): ...
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        colorSpace2: java.awt.color.ColorSpace,
        renderingHints: java.awt.RenderingHints,
    ): ...
    @typing.overload
    def __init__(
        self,
        iCC_ProfileArray: list[java.awt.color.ICC_Profile] | jpype.JArray,
        renderingHints: java.awt.RenderingHints,
    ): ...
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    @typing.overload
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    @typing.overload
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    @typing.overload
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    @typing.overload
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getICC_Profiles(self) -> typing.MutableSequence[java.awt.color.ICC_Profile]: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...

class ComponentColorModel(ColorModel):
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        boolean: bool,
        boolean2: bool,
        int: int,
        int2: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        intArray: list[int] | jpype.JArray,
        boolean: bool,
        boolean2: bool,
        int2: int,
        int3: int,
    ): ...
    def coerceData(
        self, writableRaster: WritableRaster, boolean: bool
    ) -> ColorModel: ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createCompatibleWritableRaster(self, int: int, int2: int) -> WritableRaster: ...
    def equals(self, object: typing.Any) -> bool: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    def getAlphaRaster(self, writableRaster: WritableRaster) -> WritableRaster: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    @typing.overload
    def getComponents(
        self, int: int, intArray: list[int] | jpype.JArray, int3: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, object: typing.Any, intArray: list[int] | jpype.JArray, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElement(
        self, floatArray: list[float] | jpype.JArray, int: int
    ) -> int: ...
    @typing.overload
    def getDataElement(self, intArray: list[int] | jpype.JArray, int2: int) -> int: ...
    @typing.overload
    def getDataElements(
        self, floatArray: list[float] | jpype.JArray, int: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(self, int: int, object: typing.Any) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, intArray: list[int] | jpype.JArray, int2: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getGreen(self, int: int) -> int: ...
    @typing.overload
    def getGreen(self, object: typing.Any) -> int: ...
    @typing.overload
    def getNormalizedComponents(
        self,
        intArray: list[int] | jpype.JArray,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        int3: int,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getNormalizedComponents(
        self, object: typing.Any, floatArray: list[float] | jpype.JArray, int: int
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getRGB(self, int: int) -> int: ...
    @typing.overload
    def getRGB(self, object: typing.Any) -> int: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    def getUnnormalizedComponents(
        self,
        floatArray: list[float] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
        int3: int,
    ) -> typing.MutableSequence[int]: ...
    def hashCode(self) -> int: ...
    def isCompatibleRaster(self, raster: Raster) -> bool: ...
    def isCompatibleSampleModel(self, sampleModel: SampleModel) -> bool: ...

class ComponentSampleModel(SampleModel):
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        intArray2: list[int] | jpype.JArray,
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createDataBuffer(self) -> DataBuffer: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    def equals(self, object: typing.Any) -> bool: ...
    def getBandOffsets(self) -> typing.MutableSequence[int]: ...
    def getBankIndices(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> typing.Any: ...
    def getNumDataElements(self) -> int: ...
    @typing.overload
    def getOffset(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getOffset(self, int: int, int2: int, int3: int) -> int: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getPixelStride(self) -> int: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getSample(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> int: ...
    def getSampleDouble(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    def getSampleFloat(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    @typing.overload
    def getSampleSize(self, int: int) -> int: ...
    @typing.overload
    def getSampleSize(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getScanlineStride(self) -> int: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, double: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, float: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, int4: int, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...

class ConvolveOp(BufferedImageOp, RasterOp):
    EDGE_ZERO_FILL: typing.ClassVar[int] = ...
    EDGE_NO_OP: typing.ClassVar[int] = ...
    @typing.overload
    def __init__(self, kernel: Kernel): ...
    @typing.overload
    def __init__(
        self, kernel: Kernel, int: int, renderingHints: java.awt.RenderingHints
    ): ...
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    @typing.overload
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    @typing.overload
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    @typing.overload
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    @typing.overload
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getEdgeCondition(self) -> int: ...
    def getKernel(self) -> Kernel: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...

class DataBufferByte(DataBuffer):
    @typing.overload
    def __init__(self, byteArray: list[int] | jpype.JArray | bytes, int: int): ...
    @typing.overload
    def __init__(
        self, byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ): ...
    @typing.overload
    def __init__(
        self, byteArray: list[typing.MutableSequence[int]] | jpype.JArray, int: int
    ): ...
    @typing.overload
    def __init__(
        self,
        byteArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
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

class DataBufferDouble(DataBuffer):
    @typing.overload
    def __init__(self, doubleArray: list[float] | jpype.JArray, int: int): ...
    @typing.overload
    def __init__(
        self, doubleArray: list[float] | jpype.JArray, int: int, int2: int
    ): ...
    @typing.overload
    def __init__(
        self, doubleArray: list[typing.MutableSequence[float]] | jpype.JArray, int: int
    ): ...
    @typing.overload
    def __init__(
        self,
        doubleArray: list[typing.MutableSequence[float]] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    def getData(self) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getData(self, int: int) -> typing.MutableSequence[float]: ...
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

class DataBufferFloat(DataBuffer):
    @typing.overload
    def __init__(self, floatArray: list[float] | jpype.JArray, int: int): ...
    @typing.overload
    def __init__(self, floatArray: list[float] | jpype.JArray, int: int, int2: int): ...
    @typing.overload
    def __init__(
        self, floatArray: list[typing.MutableSequence[float]] | jpype.JArray, int: int
    ): ...
    @typing.overload
    def __init__(
        self,
        floatArray: list[typing.MutableSequence[float]] | jpype.JArray,
        int: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[float]]: ...
    @typing.overload
    def getData(self) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getData(self, int: int) -> typing.MutableSequence[float]: ...
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

class DataBufferInt(DataBuffer):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    @typing.overload
    def __init__(self, intArray: list[int] | jpype.JArray, int2: int): ...
    @typing.overload
    def __init__(self, intArray: list[int] | jpype.JArray, int2: int, int3: int): ...
    @typing.overload
    def __init__(
        self, intArray: list[typing.MutableSequence[int]] | jpype.JArray, int2: int
    ): ...
    @typing.overload
    def __init__(
        self,
        intArray: list[typing.MutableSequence[int]] | jpype.JArray,
        int2: int,
        intArray2: list[int] | jpype.JArray,
    ): ...
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
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

class DataBufferShort(DataBuffer):
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
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
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

class DataBufferUShort(DataBuffer):
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
    def getBankData(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
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

class FilteredImageSource(ImageProducer):
    def __init__(self, imageProducer: ImageProducer, imageFilter: ImageFilter): ...
    def addConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def isConsumer(self, imageConsumer: ImageConsumer) -> bool: ...
    def removeConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def requestTopDownLeftRightResend(self, imageConsumer: ImageConsumer) -> None: ...
    def startProduction(self, imageConsumer: ImageConsumer) -> None: ...

class ImageFilter(ImageConsumer, java.lang.Cloneable):
    def __init__(self): ...
    def clone(self) -> typing.Any: ...
    def getFilterInstance(self, imageConsumer: ImageConsumer) -> ImageFilter: ...
    def imageComplete(self, int: int) -> None: ...
    def resendTopDownLeftRight(self, imageProducer: ImageProducer) -> None: ...
    def setColorModel(self, colorModel: ColorModel) -> None: ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    def setHints(self, int: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setProperties(
        self, hashtable: java.util.Hashtable[typing.Any, typing.Any]
    ) -> None: ...

class IndexColorModel(ColorModel):
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        byteArray: list[int] | jpype.JArray | bytes,
        byteArray2: list[int] | jpype.JArray | bytes,
        byteArray3: list[int] | jpype.JArray | bytes,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        byteArray: list[int] | jpype.JArray | bytes,
        byteArray2: list[int] | jpype.JArray | bytes,
        byteArray3: list[int] | jpype.JArray | bytes,
        byteArray4: list[int] | jpype.JArray | bytes,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        byteArray: list[int] | jpype.JArray | bytes,
        byteArray2: list[int] | jpype.JArray | bytes,
        byteArray3: list[int] | jpype.JArray | bytes,
        int3: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int3: int,
        boolean: bool,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        byteArray: list[int] | jpype.JArray | bytes,
        int3: int,
        boolean: bool,
        int4: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        int4: int,
        boolean: bool,
        int5: int,
        int6: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        int4: int,
        int5: int,
        bigInteger: java.math.BigInteger,
    ): ...
    def convertToIntDiscrete(self, raster: Raster, boolean: bool) -> BufferedImage: ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createCompatibleWritableRaster(self, int: int, int2: int) -> WritableRaster: ...
    def equals(self, object: typing.Any) -> bool: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    def getAlphas(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    def getBlues(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    @typing.overload
    def getComponentSize(self, int: int) -> int: ...
    @typing.overload
    def getComponentSize(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, int: int, intArray: list[int] | jpype.JArray, int3: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, object: typing.Any, intArray: list[int] | jpype.JArray, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElement(
        self, floatArray: list[float] | jpype.JArray, int: int
    ) -> int: ...
    @typing.overload
    def getDataElement(self, intArray: list[int] | jpype.JArray, int2: int) -> int: ...
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
    def getGreens(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def getMapSize(self) -> int: ...
    @typing.overload
    def getRGB(self, int: int) -> int: ...
    @typing.overload
    def getRGB(self, object: typing.Any) -> int: ...
    def getRGBs(self, intArray: list[int] | jpype.JArray) -> None: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    def getReds(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def getTransparency(self) -> int: ...
    def getTransparentPixel(self) -> int: ...
    def getValidPixels(self) -> java.math.BigInteger: ...
    def hashCode(self) -> int: ...
    def isCompatibleRaster(self, raster: Raster) -> bool: ...
    def isCompatibleSampleModel(self, sampleModel: SampleModel) -> bool: ...
    @typing.overload
    def isValid(self) -> bool: ...
    @typing.overload
    def isValid(self, int: int) -> bool: ...
    def toString(self) -> java.lang.String: ...

class LookupOp(BufferedImageOp, RasterOp):
    def __init__(
        self, lookupTable: LookupTable, renderingHints: java.awt.RenderingHints
    ): ...
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    @typing.overload
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    @typing.overload
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    @typing.overload
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    @typing.overload
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...
    def getTable(self) -> LookupTable: ...

class MemoryImageSource(ImageProducer):
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        int4: int,
        int5: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        int4: int,
        int5: int,
        hashtable: java.util.Hashtable[typing.Any, typing.Any],
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int3: int,
        int4: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int3: int,
        int4: int,
        hashtable: java.util.Hashtable[typing.Any, typing.Any],
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int4: int,
        int5: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int4: int,
        int5: int,
        hashtable: java.util.Hashtable[typing.Any, typing.Any],
    ): ...
    def addConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def isConsumer(self, imageConsumer: ImageConsumer) -> bool: ...
    @typing.overload
    def newPixels(
        self,
        byteArray: list[int] | jpype.JArray | bytes,
        colorModel: ColorModel,
        int: int,
        int2: int,
    ) -> None: ...
    @typing.overload
    def newPixels(self, int: int, int2: int, int3: int, int4: int) -> None: ...
    @typing.overload
    def newPixels(
        self, int: int, int2: int, int3: int, int4: int, boolean: bool
    ) -> None: ...
    @typing.overload
    def newPixels(
        self,
        intArray: list[int] | jpype.JArray,
        colorModel: ColorModel,
        int2: int,
        int3: int,
    ) -> None: ...
    @typing.overload
    def newPixels(self) -> None: ...
    def removeConsumer(self, imageConsumer: ImageConsumer) -> None: ...
    def requestTopDownLeftRightResend(self, imageConsumer: ImageConsumer) -> None: ...
    def setAnimated(self, boolean: bool) -> None: ...
    def setFullBufferUpdates(self, boolean: bool) -> None: ...
    def startProduction(self, imageConsumer: ImageConsumer) -> None: ...

class MultiPixelPackedSampleModel(SampleModel):
    @typing.overload
    def __init__(self, int: int, int2: int, int3: int, int4: int): ...
    @typing.overload
    def __init__(
        self, int: int, int2: int, int3: int, int4: int, int5: int, int6: int
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createDataBuffer(self) -> DataBuffer: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    def equals(self, object: typing.Any) -> bool: ...
    def getBitOffset(self, int: int) -> int: ...
    def getDataBitOffset(self) -> int: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> typing.Any: ...
    def getNumDataElements(self) -> int: ...
    def getOffset(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getPixelBitStride(self) -> int: ...
    def getSample(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> int: ...
    @typing.overload
    def getSampleSize(self, int: int) -> int: ...
    @typing.overload
    def getSampleSize(self) -> typing.MutableSequence[int]: ...
    def getScanlineStride(self) -> int: ...
    def getTransferType(self) -> int: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, int4: int, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, double: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, float: float, dataBuffer: DataBuffer
    ) -> None: ...

class PackedColorModel(ColorModel):
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        boolean: bool,
        int6: int,
        int7: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        int: int,
        intArray: list[int] | jpype.JArray,
        int3: int,
        boolean: bool,
        int4: int,
        int5: int,
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def equals(self, object: typing.Any) -> bool: ...
    def getAlphaRaster(self, writableRaster: WritableRaster) -> WritableRaster: ...
    def getMask(self, int: int) -> int: ...
    def getMasks(self) -> typing.MutableSequence[int]: ...
    def hashCode(self) -> int: ...
    def isCompatibleSampleModel(self, sampleModel: SampleModel) -> bool: ...

class PixelGrabber(ImageConsumer):
    @typing.overload
    def __init__(
        self,
        image: java.awt.Image,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        boolean: bool,
    ): ...
    @typing.overload
    def __init__(
        self,
        image: java.awt.Image,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ): ...
    @typing.overload
    def __init__(
        self,
        imageProducer: ImageProducer,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ): ...
    def abortGrabbing(self) -> None: ...
    def getColorModel(self) -> ColorModel: ...
    def getHeight(self) -> int: ...
    def getPixels(self) -> typing.Any: ...
    def getStatus(self) -> int: ...
    def getWidth(self) -> int: ...
    @typing.overload
    def grabPixels(self) -> bool: ...
    @typing.overload
    def grabPixels(self, long: int) -> bool: ...
    def imageComplete(self, int: int) -> None: ...
    def setColorModel(self, colorModel: ColorModel) -> None: ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    def setHints(self, int: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setProperties(
        self, hashtable: java.util.Hashtable[typing.Any, typing.Any]
    ) -> None: ...
    def startGrabbing(self) -> None: ...
    def status(self) -> int: ...

class RescaleOp(BufferedImageOp, RasterOp):
    @typing.overload
    def __init__(
        self, float: float, float2: float, renderingHints: java.awt.RenderingHints
    ): ...
    @typing.overload
    def __init__(
        self,
        floatArray: list[float] | jpype.JArray,
        floatArray2: list[float] | jpype.JArray,
        renderingHints: java.awt.RenderingHints,
    ): ...
    def createCompatibleDestImage(
        self, bufferedImage: BufferedImage, colorModel: ColorModel
    ) -> BufferedImage: ...
    def createCompatibleDestRaster(self, raster: Raster) -> WritableRaster: ...
    @typing.overload
    def filter(
        self, bufferedImage: BufferedImage, bufferedImage2: BufferedImage
    ) -> BufferedImage: ...
    @typing.overload
    def filter(
        self, raster: Raster, writableRaster: WritableRaster
    ) -> WritableRaster: ...
    @typing.overload
    def getBounds2D(
        self, bufferedImage: BufferedImage
    ) -> java.awt.geom.Rectangle2D: ...
    @typing.overload
    def getBounds2D(self, raster: Raster) -> java.awt.geom.Rectangle2D: ...
    def getNumFactors(self) -> int: ...
    def getOffsets(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...
    def getPoint2D(
        self, point2D: java.awt.geom.Point2D, point2D2: java.awt.geom.Point2D
    ) -> java.awt.geom.Point2D: ...
    def getRenderingHints(self) -> java.awt.RenderingHints: ...
    def getScaleFactors(
        self, floatArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...

class ShortLookupTable(LookupTable):
    @typing.overload
    def __init__(self, int: int, shortArray: list[int] | jpype.JArray): ...
    @typing.overload
    def __init__(
        self, int: int, shortArray: list[typing.MutableSequence[int]] | jpype.JArray
    ): ...
    def getTable(self) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
    @typing.overload
    def lookupPixel(
        self, intArray: list[int] | jpype.JArray, intArray2: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def lookupPixel(
        self,
        shortArray: list[int] | jpype.JArray,
        shortArray2: list[int] | jpype.JArray,
    ) -> typing.MutableSequence[int]: ...

class SinglePixelPackedSampleModel(SampleModel):
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(
        self, int: int, int2: int, int3: int, intArray: list[int] | jpype.JArray
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createDataBuffer(self) -> DataBuffer: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    def equals(self, object: typing.Any) -> bool: ...
    def getBitMasks(self) -> typing.MutableSequence[int]: ...
    def getBitOffsets(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> typing.Any: ...
    def getNumDataElements(self) -> int: ...
    def getOffset(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getSample(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> int: ...
    @typing.overload
    def getSampleSize(self, int: int) -> int: ...
    @typing.overload
    def getSampleSize(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getScanlineStride(self) -> int: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def setDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, double: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, float: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, int4: int, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...

class WritableRaster(Raster):
    def createWritableChild(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        int6: int,
        intArray: list[int] | jpype.JArray,
    ) -> WritableRaster: ...
    def createWritableTranslatedChild(self, int: int, int2: int) -> WritableRaster: ...
    def getWritableParent(self) -> WritableRaster: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, int3: int, int4: int, object: typing.Any
    ) -> None: ...
    @typing.overload
    def setDataElements(self, int: int, int2: int, raster: Raster) -> None: ...
    @typing.overload
    def setDataElements(self, int: int, int2: int, object: typing.Any) -> None: ...
    @typing.overload
    def setPixel(
        self, int: int, int2: int, doubleArray: list[float] | jpype.JArray
    ) -> None: ...
    @typing.overload
    def setPixel(
        self, int: int, int2: int, floatArray: list[float] | jpype.JArray
    ) -> None: ...
    @typing.overload
    def setPixel(
        self, int: int, int2: int, intArray: list[int] | jpype.JArray
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
    ) -> None: ...
    @typing.overload
    def setRect(self, int: int, int2: int, raster: Raster) -> None: ...
    @typing.overload
    def setRect(self, raster: Raster) -> None: ...
    @typing.overload
    def setSample(self, int: int, int2: int, int3: int, double: float) -> None: ...
    @typing.overload
    def setSample(self, int: int, int2: int, int3: int, float: float) -> None: ...
    @typing.overload
    def setSample(self, int: int, int2: int, int3: int, int4: int) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
    ) -> None: ...

class WritableRenderedImage(RenderedImage):
    def addTileObserver(self, tileObserver: TileObserver | typing.Callable) -> None: ...
    def getWritableTile(self, int: int, int2: int) -> WritableRaster: ...
    def getWritableTileIndices(self) -> typing.MutableSequence[java.awt.Point]: ...
    def hasTileWriters(self) -> bool: ...
    def isTileWritable(self, int: int, int2: int) -> bool: ...
    def releaseWritableTile(self, int: int, int2: int) -> None: ...
    def removeTileObserver(
        self, tileObserver: TileObserver | typing.Callable
    ) -> None: ...
    def setData(self, raster: Raster) -> None: ...

class BandedSampleModel(ComponentSampleModel):
    @typing.overload
    def __init__(self, int: int, int2: int, int3: int, int4: int): ...
    @typing.overload
    def __init__(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        intArray2: list[int] | jpype.JArray,
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createDataBuffer(self) -> DataBuffer: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    @typing.overload
    def getDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> typing.Any: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def getSample(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> int: ...
    def getSampleDouble(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    def getSampleFloat(
        self, int: int, int2: int, int3: int, dataBuffer: DataBuffer
    ) -> float: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[float]: ...
    @typing.overload
    def getSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> typing.MutableSequence[int]: ...
    def hashCode(self) -> int: ...
    @typing.overload
    def setDataElements(
        self, int: int, int2: int, object: typing.Any, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setDataElements(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        object: typing.Any,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixel(
        self,
        int: int,
        int2: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, double: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, float: float, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSample(
        self, int: int, int2: int, int3: int, int4: int, dataBuffer: DataBuffer
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        doubleArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...
    @typing.overload
    def setSamples(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        floatArray: list[float] | jpype.JArray,
        dataBuffer: DataBuffer,
    ) -> None: ...

class BaseMultiResolutionImage(AbstractMultiResolutionImage):
    @typing.overload
    def __init__(self, int: int, *image: java.awt.Image): ...
    @typing.overload
    def __init__(self, *image: java.awt.Image): ...
    def getResolutionVariant(self, double: float, double2: float) -> java.awt.Image: ...
    def getResolutionVariants(self) -> java.util.List[java.awt.Image]: ...

class BufferedImage(java.awt.Image, WritableRenderedImage, java.awt.Transparency):
    TYPE_CUSTOM: typing.ClassVar[int] = ...
    TYPE_INT_RGB: typing.ClassVar[int] = ...
    TYPE_INT_ARGB: typing.ClassVar[int] = ...
    TYPE_INT_ARGB_PRE: typing.ClassVar[int] = ...
    TYPE_INT_BGR: typing.ClassVar[int] = ...
    TYPE_3BYTE_BGR: typing.ClassVar[int] = ...
    TYPE_4BYTE_ABGR: typing.ClassVar[int] = ...
    TYPE_4BYTE_ABGR_PRE: typing.ClassVar[int] = ...
    TYPE_USHORT_565_RGB: typing.ClassVar[int] = ...
    TYPE_USHORT_555_RGB: typing.ClassVar[int] = ...
    TYPE_BYTE_GRAY: typing.ClassVar[int] = ...
    TYPE_USHORT_GRAY: typing.ClassVar[int] = ...
    TYPE_BYTE_BINARY: typing.ClassVar[int] = ...
    TYPE_BYTE_INDEXED: typing.ClassVar[int] = ...
    @typing.overload
    def __init__(self, int: int, int2: int, int3: int): ...
    @typing.overload
    def __init__(
        self, int: int, int2: int, int3: int, indexColorModel: IndexColorModel
    ): ...
    @typing.overload
    def __init__(
        self,
        colorModel: ColorModel,
        writableRaster: WritableRaster,
        boolean: bool,
        hashtable: java.util.Hashtable[typing.Any, typing.Any],
    ): ...
    def addTileObserver(self, tileObserver: TileObserver | typing.Callable) -> None: ...
    def coerceData(self, boolean: bool) -> None: ...
    def copyData(self, writableRaster: WritableRaster) -> WritableRaster: ...
    def createGraphics(self) -> java.awt.Graphics2D: ...
    def getAlphaRaster(self) -> WritableRaster: ...
    def getColorModel(self) -> ColorModel: ...
    @typing.overload
    def getData(self) -> Raster: ...
    @typing.overload
    def getData(self, rectangle: java.awt.Rectangle) -> Raster: ...
    def getGraphics(self) -> java.awt.Graphics: ...
    @typing.overload
    def getHeight(self) -> int: ...
    @typing.overload
    def getHeight(self, imageObserver: ImageObserver | typing.Callable) -> int: ...
    def getMinTileX(self) -> int: ...
    def getMinTileY(self) -> int: ...
    def getMinX(self) -> int: ...
    def getMinY(self) -> int: ...
    def getNumXTiles(self) -> int: ...
    def getNumYTiles(self) -> int: ...
    @typing.overload
    def getProperty(self, string: java.lang.String | str) -> typing.Any: ...
    @typing.overload
    def getProperty(
        self,
        string: java.lang.String | str,
        imageObserver: ImageObserver | typing.Callable,
    ) -> typing.Any: ...
    def getPropertyNames(self) -> typing.MutableSequence[java.lang.String]: ...
    @typing.overload
    def getRGB(self, int: int, int2: int) -> int: ...
    @typing.overload
    def getRGB(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> typing.MutableSequence[int]: ...
    def getRaster(self) -> WritableRaster: ...
    def getSampleModel(self) -> SampleModel: ...
    def getSource(self) -> ImageProducer: ...
    def getSources(self) -> java.util.Vector[RenderedImage]: ...
    def getSubimage(
        self, int: int, int2: int, int3: int, int4: int
    ) -> BufferedImage: ...
    def getTile(self, int: int, int2: int) -> Raster: ...
    def getTileGridXOffset(self) -> int: ...
    def getTileGridYOffset(self) -> int: ...
    def getTileHeight(self) -> int: ...
    def getTileWidth(self) -> int: ...
    def getTransparency(self) -> int: ...
    def getType(self) -> int: ...
    @typing.overload
    def getWidth(self) -> int: ...
    @typing.overload
    def getWidth(self, imageObserver: ImageObserver | typing.Callable) -> int: ...
    def getWritableTile(self, int: int, int2: int) -> WritableRaster: ...
    def getWritableTileIndices(self) -> typing.MutableSequence[java.awt.Point]: ...
    def hasTileWriters(self) -> bool: ...
    def isAlphaPremultiplied(self) -> bool: ...
    def isTileWritable(self, int: int, int2: int) -> bool: ...
    def releaseWritableTile(self, int: int, int2: int) -> None: ...
    def removeTileObserver(
        self, tileObserver: TileObserver | typing.Callable
    ) -> None: ...
    def setData(self, raster: Raster) -> None: ...
    @typing.overload
    def setRGB(self, int: int, int2: int, int3: int) -> None: ...
    @typing.overload
    def setRGB(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def toString(self) -> java.lang.String: ...

class BufferedImageFilter(ImageFilter, java.lang.Cloneable):
    def __init__(self, bufferedImageOp: BufferedImageOp): ...
    def getBufferedImageOp(self) -> BufferedImageOp: ...
    def imageComplete(self, int: int) -> None: ...
    def setColorModel(self, colorModel: ColorModel) -> None: ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...

class CropImageFilter(ImageFilter):
    def __init__(self, int: int, int2: int, int3: int, int4: int): ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setProperties(
        self, hashtable: java.util.Hashtable[typing.Any, typing.Any]
    ) -> None: ...

class DirectColorModel(PackedColorModel):
    @typing.overload
    def __init__(self, int: int, int2: int, int3: int, int4: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int, int3: int, int4: int, int5: int): ...
    @typing.overload
    def __init__(
        self,
        colorSpace: java.awt.color.ColorSpace,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        boolean: bool,
        int6: int,
    ): ...
    def coerceData(
        self, writableRaster: WritableRaster, boolean: bool
    ) -> ColorModel: ...
    def createCompatibleWritableRaster(self, int: int, int2: int) -> WritableRaster: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    def getAlphaMask(self) -> int: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    def getBlueMask(self) -> int: ...
    @typing.overload
    def getComponents(
        self, int: int, intArray: list[int] | jpype.JArray, int3: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getComponents(
        self, object: typing.Any, intArray: list[int] | jpype.JArray, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getDataElement(
        self, floatArray: list[float] | jpype.JArray, int: int
    ) -> int: ...
    @typing.overload
    def getDataElement(self, intArray: list[int] | jpype.JArray, int2: int) -> int: ...
    @typing.overload
    def getDataElements(
        self, floatArray: list[float] | jpype.JArray, int: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getDataElements(self, int: int, object: typing.Any) -> typing.Any: ...
    @typing.overload
    def getDataElements(
        self, intArray: list[int] | jpype.JArray, int2: int, object: typing.Any
    ) -> typing.Any: ...
    @typing.overload
    def getGreen(self, int: int) -> int: ...
    @typing.overload
    def getGreen(self, object: typing.Any) -> int: ...
    def getGreenMask(self) -> int: ...
    @typing.overload
    def getRGB(self, int: int) -> int: ...
    @typing.overload
    def getRGB(self, object: typing.Any) -> int: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    def getRedMask(self) -> int: ...
    def isCompatibleRaster(self, raster: Raster) -> bool: ...
    def toString(self) -> java.lang.String: ...

class PixelInterleavedSampleModel(ComponentSampleModel):
    def __init__(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
        intArray: list[int] | jpype.JArray,
    ): ...
    def createCompatibleSampleModel(self, int: int, int2: int) -> SampleModel: ...
    def createSubsetSampleModel(
        self, intArray: list[int] | jpype.JArray
    ) -> SampleModel: ...
    def hashCode(self) -> int: ...

class RGBImageFilter(ImageFilter):
    def filterIndexColorModel(
        self, indexColorModel: IndexColorModel
    ) -> IndexColorModel: ...
    def filterRGB(self, int: int, int2: int, int3: int) -> int: ...
    def filterRGBPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setColorModel(self, colorModel: ColorModel) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def substituteColorModel(
        self, colorModel: ColorModel, colorModel2: ColorModel
    ) -> None: ...

class ReplicateScaleFilter(ImageFilter):
    def __init__(self, int: int, int2: int): ...
    def setDimensions(self, int: int, int2: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...
    def setProperties(
        self, hashtable: java.util.Hashtable[typing.Any, typing.Any]
    ) -> None: ...

class AreaAveragingScaleFilter(ReplicateScaleFilter):
    def __init__(self, int: int, int2: int): ...
    def setHints(self, int: int) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        byteArray: list[int] | jpype.JArray | bytes,
        int5: int,
        int6: int,
    ) -> None: ...
    @typing.overload
    def setPixels(
        self,
        int: int,
        int2: int,
        int3: int,
        int4: int,
        colorModel: ColorModel,
        intArray: list[int] | jpype.JArray,
        int6: int,
        int7: int,
    ) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.awt.image")``.

    AbstractMultiResolutionImage: type[AbstractMultiResolutionImage]
    AffineTransformOp: type[AffineTransformOp]
    AreaAveragingScaleFilter: type[AreaAveragingScaleFilter]
    BandCombineOp: type[BandCombineOp]
    BandedSampleModel: type[BandedSampleModel]
    BaseMultiResolutionImage: type[BaseMultiResolutionImage]
    BufferStrategy: type[BufferStrategy]
    BufferedImage: type[BufferedImage]
    BufferedImageFilter: type[BufferedImageFilter]
    BufferedImageOp: type[BufferedImageOp]
    ByteLookupTable: type[ByteLookupTable]
    ColorConvertOp: type[ColorConvertOp]
    ColorModel: type[ColorModel]
    ComponentColorModel: type[ComponentColorModel]
    ComponentSampleModel: type[ComponentSampleModel]
    ConvolveOp: type[ConvolveOp]
    CropImageFilter: type[CropImageFilter]
    DataBuffer: type[DataBuffer]
    DataBufferByte: type[DataBufferByte]
    DataBufferDouble: type[DataBufferDouble]
    DataBufferFloat: type[DataBufferFloat]
    DataBufferInt: type[DataBufferInt]
    DataBufferShort: type[DataBufferShort]
    DataBufferUShort: type[DataBufferUShort]
    DirectColorModel: type[DirectColorModel]
    FilteredImageSource: type[FilteredImageSource]
    ImageConsumer: type[ImageConsumer]
    ImageFilter: type[ImageFilter]
    ImageObserver: type[ImageObserver]
    ImageProducer: type[ImageProducer]
    ImagingOpException: type[ImagingOpException]
    IndexColorModel: type[IndexColorModel]
    Kernel: type[Kernel]
    LookupOp: type[LookupOp]
    LookupTable: type[LookupTable]
    MemoryImageSource: type[MemoryImageSource]
    MultiPixelPackedSampleModel: type[MultiPixelPackedSampleModel]
    MultiResolutionImage: type[MultiResolutionImage]
    PackedColorModel: type[PackedColorModel]
    PixelGrabber: type[PixelGrabber]
    PixelInterleavedSampleModel: type[PixelInterleavedSampleModel]
    RGBImageFilter: type[RGBImageFilter]
    Raster: type[Raster]
    RasterFormatException: type[RasterFormatException]
    RasterOp: type[RasterOp]
    RenderedImage: type[RenderedImage]
    ReplicateScaleFilter: type[ReplicateScaleFilter]
    RescaleOp: type[RescaleOp]
    SampleModel: type[SampleModel]
    ShortLookupTable: type[ShortLookupTable]
    SinglePixelPackedSampleModel: type[SinglePixelPackedSampleModel]
    TileObserver: type[TileObserver]
    VolatileImage: type[VolatileImage]
    WritableRaster: type[WritableRaster]
    WritableRenderedImage: type[WritableRenderedImage]
    renderable: java.awt.image.renderable.__module_protocol__
