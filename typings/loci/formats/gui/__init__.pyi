import typing
from typing import Protocol

import java.awt
import java.awt.color
import java.awt.event
import java.awt.image
import java.beans
import java.io
import java.lang
import java.util
import javax.swing
import javax.swing.event
import javax.swing.filechooser
import javax.swing.tree
import jpype
import jpype.protocol
import loci.formats
import loci.formats.cache
import loci.formats.meta
import org.w3c.dom

class AWTImageTools:
    @typing.overload
    @staticmethod
    def autoscale(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def autoscale(
        bufferedImage: java.awt.image.BufferedImage, int: int, int2: int
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def blankImage(
        int: int, int2: int, int3: int, int4: int
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
    @staticmethod
    def convertRenderedImage(
        renderedImage: java.awt.image.RenderedImage,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def copyScaled(
        bufferedImage: java.awt.image.BufferedImage,
        bufferedImage2: java.awt.image.BufferedImage,
        object: typing.Any,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def get8BitLookupTable(
        colorModel: java.awt.image.ColorModel,
    ) -> typing.MutableSequence[typing.MutableSequence[int]]: ...
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
    @staticmethod
    def getDefaultConfiguration() -> java.awt.GraphicsConfiguration: ...
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
    @staticmethod
    def getLookupTable(
        colorModel: java.awt.image.ColorModel,
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
    @staticmethod
    def getPixelType(bufferedImage: java.awt.image.BufferedImage) -> int: ...
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
    def getSize(image: java.awt.Image) -> java.awt.Dimension: ...
    @staticmethod
    def getSubimage(
        bufferedImage: java.awt.image.BufferedImage,
        boolean: bool,
        int: int,
        int2: int,
        int3: int,
        int4: int,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def indexedToRGB(
        bufferedImage: java.awt.image.BufferedImage, boolean: bool
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def loadImage(image: java.awt.Image) -> bool: ...
    @typing.overload
    @staticmethod
    def makeBuffered(image: java.awt.Image) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeBuffered(
        image: java.awt.Image, colorModel: java.awt.image.ColorModel
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def makeColorModel(int: int, int2: int) -> java.awt.image.ColorModel: ...
    @staticmethod
    def makeColorSpace(int: int) -> java.awt.color.ColorSpace: ...
    @staticmethod
    def makeCompatible(
        bufferedImage: java.awt.image.BufferedImage,
        graphicsConfiguration: java.awt.GraphicsConfiguration,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def makeImage(
        byteArray: list[int] | jpype.JArray | bytes,
        boolean: bool,
        metadataRetrieve: loci.formats.meta.MetadataRetrieve,
        int: int,
    ) -> java.awt.image.BufferedImage: ...
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
    @staticmethod
    def makeUnsigned(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def mergeChannels(
        bufferedImageArray: list[java.awt.image.BufferedImage] | jpype.JArray,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def openImage(
        byteArray: list[int] | jpype.JArray | bytes,
        iFormatReader: loci.formats.IFormatReader,
        int: int,
        int2: int,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def openImage(
        byteArray: list[int] | jpype.JArray | bytes,
        iFormatReader: loci.formats.IFormatReader,
        int: int,
        int2: int,
        boolean: bool,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def padImage(
        bufferedImage: java.awt.image.BufferedImage, int: int, int2: int
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def scale(
        bufferedImage: java.awt.image.BufferedImage, int: int, int2: int, boolean: bool
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def scale2D(
        bufferedImage: java.awt.image.BufferedImage,
        int: int,
        int2: int,
        object: typing.Any,
        graphicsConfiguration: java.awt.GraphicsConfiguration,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    @staticmethod
    def scale2D(
        bufferedImage: java.awt.image.BufferedImage,
        int: int,
        int2: int,
        object: typing.Any,
        colorModel: java.awt.image.ColorModel,
    ) -> java.awt.image.BufferedImage: ...
    @staticmethod
    def scaleAWT(
        bufferedImage: java.awt.image.BufferedImage, int: int, int2: int, int3: int
    ) -> java.awt.Image: ...
    @staticmethod
    def splitChannels(
        bufferedImage: java.awt.image.BufferedImage,
    ) -> typing.MutableSequence[java.awt.image.BufferedImage]: ...

class BufferedImageReader(loci.formats.ReaderWrapper):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, iFormatReader: loci.formats.IFormatReader): ...
    @staticmethod
    def makeBufferedImageReader(
        iFormatReader: loci.formats.IFormatReader,
    ) -> BufferedImageReader: ...
    @typing.overload
    def openImage(self, int: int) -> java.awt.image.BufferedImage: ...
    @typing.overload
    def openImage(
        self, int: int, int2: int, int3: int, int4: int, int5: int
    ) -> java.awt.image.BufferedImage: ...
    def openThumbImage(self, int: int) -> java.awt.image.BufferedImage: ...

class BufferedImageSource(loci.formats.cache.ICacheSource):
    def __init__(self, iFormatReader: loci.formats.IFormatReader): ...
    def getObject(self, int: int) -> typing.Any: ...
    def getObjectCount(self) -> int: ...

class BufferedImageWriter(loci.formats.WriterWrapper):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, iFormatWriter: loci.formats.IFormatWriter): ...
    @staticmethod
    def makeBufferedImageWriter(
        iFormatWriter: loci.formats.IFormatWriter,
    ) -> BufferedImageWriter: ...
    @typing.overload
    def saveImage(
        self, int: int, bufferedImage: java.awt.image.BufferedImage
    ) -> None: ...
    @typing.overload
    def saveImage(
        self,
        int: int,
        bufferedImage: java.awt.image.BufferedImage,
        int2: int,
        int3: int,
        int4: int,
        int5: int,
    ) -> None: ...
    @staticmethod
    def toBytes(
        bufferedImage: java.awt.image.BufferedImage,
        iFormatWriter: loci.formats.IFormatWriter,
    ) -> typing.MutableSequence[int]: ...

class CacheComponent(
    javax.swing.JPanel,
    java.awt.event.ActionListener,
    loci.formats.cache.CacheListener,
    javax.swing.event.ChangeListener,
):
    @typing.overload
    def __init__(
        self,
        cache: loci.formats.cache.Cache,
        stringArray: list[java.lang.String] | jpype.JArray,
    ): ...
    @typing.overload
    def __init__(
        self,
        cache: loci.formats.cache.Cache,
        stringArray: list[java.lang.String] | jpype.JArray,
        string2: java.lang.String | str,
    ): ...
    def actionPerformed(self, actionEvent: java.awt.event.ActionEvent) -> None: ...
    def cacheUpdated(self, cacheEvent: loci.formats.cache.CacheEvent) -> None: ...
    def dispose(self) -> None: ...
    def getCache(self) -> loci.formats.cache.Cache: ...
    def stateChanged(self, changeEvent: javax.swing.event.ChangeEvent) -> None: ...

class CacheIndicator(javax.swing.JComponent, loci.formats.cache.CacheListener):
    @typing.overload
    def __init__(self, cache: loci.formats.cache.Cache, int: int): ...
    @typing.overload
    def __init__(
        self,
        cache: loci.formats.cache.Cache,
        int: int,
        component: java.awt.Component,
        int2: int,
        int3: int,
    ): ...
    def cacheUpdated(self, cacheEvent: loci.formats.cache.CacheEvent) -> None: ...
    def getMaximumSize(self) -> java.awt.Dimension: ...
    def getMinimumSize(self) -> java.awt.Dimension: ...
    def getPreferredSize(self) -> java.awt.Dimension: ...
    def paintComponent(self, graphics: java.awt.Graphics) -> None: ...

class ComboFileFilter(
    javax.swing.filechooser.FileFilter, java.io.FileFilter, java.lang.Comparable
):
    def __init__(
        self,
        fileFilterArray: list[javax.swing.filechooser.FileFilter] | jpype.JArray,
        string: java.lang.String | str,
    ): ...
    def accept(self, file: java.io.File | jpype.protocol.SupportsPath) -> bool: ...
    def compareTo(self, object: typing.Any) -> int: ...
    def getDescription(self) -> java.lang.String: ...
    def getFilters(
        self,
    ) -> typing.MutableSequence[javax.swing.filechooser.FileFilter]: ...
    @typing.overload
    @staticmethod
    def sortFilters(
        vector: java.util.Vector,
    ) -> typing.MutableSequence[javax.swing.filechooser.FileFilter]: ...
    @typing.overload
    @staticmethod
    def sortFilters(
        fileFilterArray: list[javax.swing.filechooser.FileFilter] | jpype.JArray,
    ) -> typing.MutableSequence[javax.swing.filechooser.FileFilter]: ...
    def toString(self) -> java.lang.String: ...

class DataConverter(
    javax.swing.JFrame,
    java.awt.event.ActionListener,
    javax.swing.event.ChangeListener,
    java.lang.Runnable,
):
    def __init__(self): ...
    def actionPerformed(self, actionEvent: java.awt.event.ActionEvent) -> None: ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def run(self) -> None: ...
    def stateChanged(self, changeEvent: javax.swing.event.ChangeEvent) -> None: ...

class ExtensionFileFilter(
    javax.swing.filechooser.FileFilter, java.io.FileFilter, java.lang.Comparable
):
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ): ...
    @typing.overload
    def __init__(
        self,
        stringArray: list[java.lang.String] | jpype.JArray,
        string2: java.lang.String | str,
    ): ...
    def accept(self, file: java.io.File | jpype.protocol.SupportsPath) -> bool: ...
    def compareTo(self, object: typing.Any) -> int: ...
    def getDescription(self) -> java.lang.String: ...
    def getExtension(self) -> java.lang.String: ...
    def getExtensions(self) -> typing.MutableSequence[java.lang.String]: ...
    def toString(self) -> java.lang.String: ...

class FormatFileFilter(
    javax.swing.filechooser.FileFilter, java.io.FileFilter, java.lang.Comparable
):
    @typing.overload
    def __init__(self, iFormatReader: loci.formats.IFormatReader): ...
    @typing.overload
    def __init__(self, iFormatReader: loci.formats.IFormatReader, boolean: bool): ...
    def accept(self, file: java.io.File | jpype.protocol.SupportsPath) -> bool: ...
    def compareTo(self, object: typing.Any) -> int: ...
    def getDescription(self) -> java.lang.String: ...
    def getReader(self) -> loci.formats.IFormatReader: ...
    def toString(self) -> java.lang.String: ...

class GUITools:
    @typing.overload
    @staticmethod
    def buildFileChooser(
        fileFilterArray: list[javax.swing.filechooser.FileFilter] | jpype.JArray,
    ) -> javax.swing.JFileChooser: ...
    @typing.overload
    @staticmethod
    def buildFileChooser(
        fileFilterArray: list[javax.swing.filechooser.FileFilter] | jpype.JArray,
        boolean: bool,
    ) -> javax.swing.JFileChooser: ...
    @typing.overload
    @staticmethod
    def buildFileChooser(
        iFormatHandler: loci.formats.IFormatHandler,
    ) -> javax.swing.JFileChooser: ...
    @typing.overload
    @staticmethod
    def buildFileChooser(
        iFormatHandler: loci.formats.IFormatHandler, boolean: bool
    ) -> javax.swing.JFileChooser: ...
    @staticmethod
    def buildFileFilters(
        iFormatHandler: loci.formats.IFormatHandler,
    ) -> typing.MutableSequence[javax.swing.filechooser.FileFilter]: ...

class ImageViewer(
    javax.swing.JFrame,
    java.awt.event.ActionListener,
    javax.swing.event.ChangeListener,
    java.awt.event.KeyListener,
    java.awt.event.MouseMotionListener,
    java.lang.Runnable,
    java.awt.event.WindowListener,
):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, boolean: bool): ...
    def actionPerformed(self, actionEvent: java.awt.event.ActionEvent) -> None: ...
    def getC(self) -> int: ...
    def getImage(self) -> java.awt.image.BufferedImage: ...
    def getImageIndex(self) -> int: ...
    def getT(self) -> int: ...
    def getZ(self) -> int: ...
    def keyPressed(self, keyEvent: java.awt.event.KeyEvent) -> None: ...
    def keyReleased(self, keyEvent: java.awt.event.KeyEvent) -> None: ...
    def keyTyped(self, keyEvent: java.awt.event.KeyEvent) -> None: ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def mouseDragged(self, mouseEvent: java.awt.event.MouseEvent) -> None: ...
    def mouseMoved(self, mouseEvent: java.awt.event.MouseEvent) -> None: ...
    def open(self, string: java.lang.String | str) -> None: ...
    def run(self) -> None: ...
    def save(self, string: java.lang.String | str) -> None: ...
    @typing.overload
    def setImages(
        self, bufferedImageArray: list[java.awt.image.BufferedImage] | jpype.JArray
    ) -> None: ...
    @typing.overload
    def setImages(
        self,
        iFormatReader: loci.formats.IFormatReader,
        bufferedImageArray: list[java.awt.image.BufferedImage] | jpype.JArray,
    ) -> None: ...
    def setVisible(self, boolean: bool) -> None: ...
    def stateChanged(self, changeEvent: javax.swing.event.ChangeEvent) -> None: ...
    def windowActivated(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowClosed(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowClosing(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowDeactivated(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowDeiconified(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowIconified(self, windowEvent: java.awt.event.WindowEvent) -> None: ...
    def windowOpened(self, windowEvent: java.awt.event.WindowEvent) -> None: ...

class Index16ColorModel(java.awt.image.ColorModel):
    def __init__(
        self,
        int: int,
        int2: int,
        shortArray: list[typing.MutableSequence[int]] | jpype.JArray,
        boolean: bool,
    ): ...
    def createCompatibleWritableRaster(
        self, int: int, int2: int
    ) -> java.awt.image.WritableRaster: ...
    @typing.overload
    def getAlpha(self, object: typing.Any) -> int: ...
    @typing.overload
    def getAlpha(self, int: int) -> int: ...
    def getAlphas(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getBlue(self, object: typing.Any) -> int: ...
    @typing.overload
    def getBlue(self, int: int) -> int: ...
    def getBlues(self) -> typing.MutableSequence[int]: ...
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
    def getGreen(self, object: typing.Any) -> int: ...
    @typing.overload
    def getGreen(self, int: int) -> int: ...
    def getGreens(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def getRed(self, object: typing.Any) -> int: ...
    @typing.overload
    def getRed(self, int: int) -> int: ...
    def getReds(self) -> typing.MutableSequence[int]: ...
    def isCompatibleRaster(self, raster: java.awt.image.Raster) -> bool: ...

class NoExtensionFileFilter(
    javax.swing.filechooser.FileFilter, java.io.FileFilter, java.lang.Comparable
):
    def __init__(self): ...
    def accept(self, file: java.io.File | jpype.protocol.SupportsPath) -> bool: ...
    def compareTo(self, object: typing.Any) -> int: ...
    def getDescription(self) -> java.lang.String: ...
    def toString(self) -> java.lang.String: ...

class PreviewPane(
    javax.swing.JPanel, java.beans.PropertyChangeListener, java.lang.Runnable
):
    def __init__(self, jFileChooser: javax.swing.JFileChooser): ...
    def close(self) -> None: ...
    def getPreferredSize(self) -> java.awt.Dimension: ...
    def propertyChange(
        self, propertyChangeEvent: java.beans.PropertyChangeEvent
    ) -> None: ...
    def run(self) -> None: ...

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

class SignedColorModel(java.awt.image.ColorModel):
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

class XMLCellRenderer(javax.swing.tree.DefaultTreeCellRenderer):
    def __init__(self): ...
    def getTreeCellRendererComponent(
        self,
        jTree: javax.swing.JTree,
        object: typing.Any,
        boolean: bool,
        boolean2: bool,
        boolean3: bool,
        int: int,
        boolean4: bool,
    ) -> java.awt.Component: ...
    @staticmethod
    def makeJTree(document: org.w3c.dom.Document) -> javax.swing.JTree: ...

class XMLWindow(javax.swing.JFrame):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    def getDocument(self) -> org.w3c.dom.Document: ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    def setDocument(self, document: org.w3c.dom.Document) -> None: ...
    @typing.overload
    def setXML(self, file: java.io.File | jpype.protocol.SupportsPath) -> None: ...
    @typing.overload
    def setXML(self, string: java.lang.String | str) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.formats.gui")``.

    AWTImageTools: type[AWTImageTools]
    BufferedImageReader: type[BufferedImageReader]
    BufferedImageSource: type[BufferedImageSource]
    BufferedImageWriter: type[BufferedImageWriter]
    CacheComponent: type[CacheComponent]
    CacheIndicator: type[CacheIndicator]
    ComboFileFilter: type[ComboFileFilter]
    DataConverter: type[DataConverter]
    ExtensionFileFilter: type[ExtensionFileFilter]
    FormatFileFilter: type[FormatFileFilter]
    GUITools: type[GUITools]
    ImageViewer: type[ImageViewer]
    Index16ColorModel: type[Index16ColorModel]
    NoExtensionFileFilter: type[NoExtensionFileFilter]
    PreviewPane: type[PreviewPane]
    SignedByteBuffer: type[SignedByteBuffer]
    SignedColorModel: type[SignedColorModel]
    SignedShortBuffer: type[SignedShortBuffer]
    TwoChannelColorSpace: type[TwoChannelColorSpace]
    UnsignedIntBuffer: type[UnsignedIntBuffer]
    UnsignedIntColorModel: type[UnsignedIntColorModel]
    XMLCellRenderer: type[XMLCellRenderer]
    XMLWindow: type[XMLWindow]
