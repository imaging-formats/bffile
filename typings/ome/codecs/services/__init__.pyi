import typing
from typing import Protocol

import java.awt.image
import java.io
import java.lang
import loci.common.services
import ome.codecs

class JAIIIOService(loci.common.services.Service):
    @typing.overload
    def readImage(
        self, inputStream: java.io.InputStream
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    def readImage(
        self,
        inputStream: java.io.InputStream,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    def readRaster(self, inputStream: java.io.InputStream) -> java.awt.image.Raster: ...
    @typing.overload
    def readRaster(
        self,
        inputStream: java.io.InputStream,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> java.awt.image.Raster: ...
    def writeImage(
        self,
        outputStream: java.io.OutputStream,
        bufferedImage: java.awt.image.BufferedImage,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> None: ...

class JAIIIOServiceImpl(loci.common.services.AbstractService, JAIIIOService):
    NO_J2K_MSG: typing.ClassVar[java.lang.String] = ...
    def __init__(self): ...
    @typing.overload
    def readImage(
        self, inputStream: java.io.InputStream
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    def readImage(
        self,
        inputStream: java.io.InputStream,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> java.awt.image.BufferedImage: ...
    @typing.overload
    def readRaster(self, inputStream: java.io.InputStream) -> java.awt.image.Raster: ...
    @typing.overload
    def readRaster(
        self,
        inputStream: java.io.InputStream,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> java.awt.image.Raster: ...
    def writeImage(
        self,
        outputStream: java.io.OutputStream,
        bufferedImage: java.awt.image.BufferedImage,
        jPEG2000CodecOptions: ome.codecs.JPEG2000CodecOptions,
    ) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.codecs.services")``.

    JAIIIOService: type[JAIIIOService]
    JAIIIOServiceImpl: type[JAIIIOServiceImpl]
