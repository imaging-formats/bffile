from __future__ import annotations
from typing import overload, Any, List, Optional, Dict

# from loci.common import RandomAccessInputStream
# from loci.formats.meta import MetadataStore
# from loci.formats import (
#     IFormatHandler,
#     IPyramidHandler,
#     ICompressedTileReader,
#     FormatException,
#     FileInfo,
#     CoreMetadata,
#     Modulo,
# )


class IFormatReader(IFormatHandler, IPyramidHandler, ICompressedTileReader):
    """
    Interface for all biological file format readers.
    """
    MUST_GROUP: int = ...
    CAN_GROUP: int = ...
    CANNOT_GROUP: int = ...

    @overload
    def isThisType(self, name: str, open: bool) -> bool: ...
    @overload
    def isThisType(self, block: bytes) -> bool: ...
    @overload
    def isThisType(self, stream: Any) -> bool: ...
    def isThisType(self, *args: Any) -> bool:
        """
        Checks if the given file is a valid instance of this file format.

        If name and open flag are provided, the file may be opened for further analysis.
        If a block of bytes is provided, checks header validity.
        If a RandomAccessInputStream is provided, checks the stream content.
        """
    def getImageCount(self) -> int:
        """Determines the number of image planes in the current file."""

    def isRGB(self) -> bool:
        """Checks if the image planes have more than one channel per openBytes call."""

    def getSizeX(self) -> int:
        """Gets the size of the X dimension."""

    def getSizeY(self) -> int:
        """Gets the size of the Y dimension."""

    def getSizeZ(self) -> int:
        """Gets the size of the Z dimension."""

    def getSizeC(self) -> int:
        """Gets the size of the C dimension."""

    def getSizeT(self) -> int:
        """Gets the size of the T dimension."""

    def getPixelType(self) -> int:
        """Gets the pixel type as defined in FormatTools (e.g., INT8)."""

    def getBitsPerPixel(self) -> int:
        """Gets the number of valid bits per pixel."""

    def getEffectiveSizeC(self) -> int:
        """Gets the effective number of channels, accounting for RGB interleaving."""

    def getRGBChannelCount(self) -> int:
        """Gets the number of channels returned per openBytes call."""

    def isIndexed(self) -> bool:
        """Checks if the image planes are indexed color."""

    def isFalseColor(self) -> bool:
        """Returns true if indexed color data is for visualization only."""

    def get8BitLookupTable(self) -> List[bytes]:
        """Gets the 8-bit color lookup table for the current image plane.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If reading the file fails.
        """

    def get16BitLookupTable(self) -> List[List[int]]:
        """Gets the 16-bit color lookup table for the current image plane.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If reading the file fails.
        """

    def getModuloZ(self) -> Modulo:
        """Gets the Z modulo definition."""

    def getModuloC(self) -> Modulo:
        """Gets the C modulo definition."""

    def getModuloT(self) -> Modulo:
        """Gets the T modulo definition."""

    def getThumbSizeX(self) -> int:
        """Gets the X dimension size for the thumbnail."""

    def getThumbSizeY(self) -> int:
        """Gets the Y dimension size for the thumbnail."""

    def isLittleEndian(self) -> bool:
        """Checks if the data is in little-endian format."""

    def getDimensionOrder(self) -> str:
        """Gets the five-character dimension order (e.g., 'XYCTZ')."""

    def isOrderCertain(self) -> bool:
        """Determines if the dimension order and sizes are known with certainty."""

    def isThumbnailSeries(self) -> bool:
        """Checks if the current series is a lower-resolution thumbnail."""

    @overload
    def isInterleaved(self) -> bool: ...
    @overload
    def isInterleaved(self, subC: int) -> bool: ...
    def isInterleaved(self, *args: Any) -> bool:
        """Checks if channel data is interleaved (optionally for a sub-channel)."""

    @overload
    def openBytes(self, no: int) -> bytes: ...
    @overload
    def openBytes(self, no: int, x: int, y: int, w: int, h: int) -> bytes: ...
    @overload
    def openBytes(self, no: int, buf: bytearray) -> bytearray: ...
    @overload
    def openBytes(
        self,
        no: int,
        buf: bytearray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> bytearray: ...
    def openBytes(self, *args: Any) -> Any:
        """Obtains image plane or sub-image data as bytes or fills a buffer.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If reading the file fails.
        """

    def openPlane(
        self, no: int, x: int, y: int, w: int, h: int
    ) -> Any:
        """Obtains the specified plane in the reader's native format."""

    def openThumbBytes(self, no: int) -> bytes:
        """Obtains a thumbnail for the specified image plane."""

    def close(self, fileOnly: bool) -> None:
        """Closes the currently open file (fileOnly: if True, only file stream)."""

    def getSeriesCount(self) -> int:
        """Gets the number of series in the file."""

    def setSeries(self, no: int) -> None:
        """Activates the specified series and resets resolution to 0."""

    def getSeries(self) -> int:
        """Gets the currently active series."""

    def setNormalized(self, normalize: bool) -> None:
        """Specifies whether to normalize float data."""

    def isNormalized(self) -> bool:
        """Checks if float data normalization is enabled."""

    def setOriginalMetadataPopulated(self, populate: bool) -> None:
        """Specifies whether to save proprietary metadata."""

    def isOriginalMetadataPopulated(self) -> bool:
        """Checks if proprietary metadata is saved in the MetadataStore."""

    def setGroupFiles(self, group: bool) -> None:
        """Specifies whether to group files in multi-file formats."""

    def isGroupFiles(self) -> bool:
        """Checks if file grouping is enabled for multi-file formats."""

    def setFillColor(self, color: Optional[int]) -> None:
        """Sets the fill value for undefined pixels (no-op by default)."""

    def getFillColor(self) -> Optional[int]:
        """Returns the fill value for undefined pixels (default 0)."""

    def isMetadataComplete(self) -> bool:
        """Checks if the format's metadata is completely parsed."""

    def fileGroupOption(self, id: str) -> int:
        """Indicates grouping requirements for multi-file datasets.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If file operations fail.
        """

    @overload
    def getUsedFiles(self) -> List[str]: ...
    @overload
    def getUsedFiles(self, noPixels: bool) -> List[str]: ...
    def getUsedFiles(self, noPixels: bool = ...) -> List[str]:
        """Returns filenames needed to open this dataset, optionally excluding pixel files."""

    @overload
    def getSeriesUsedFiles(self) -> List[str]: ...
    @overload
    def getSeriesUsedFiles(self, noPixels: bool) -> List[str]: ...
    def getSeriesUsedFiles(self, noPixels: bool = ...) -> List[str]:
        """Returns filenames needed to open the current series, optionally excluding pixel files."""

    def getAdvancedUsedFiles(self, noPixels: bool) -> List[FileInfo]:
        """Returns FileInfo objects for files needed to open this dataset."""

    def getAdvancedSeriesUsedFiles(self, noPixels: bool) -> List[FileInfo]:
        """Returns FileInfo objects for files needed to open the current series."""

    def getCurrentFile(self) -> str:
        """Returns the path of the current file."""

    def getDomains(self) -> List[str]:
        """Returns the list of domains represented by the current file."""

    @overload
    def getIndex(self, z: int, c: int, t: int) -> int: ...
    @overload
    def getIndex(
        self, z: int, c: int, t: int, moduloZ: int, moduloC: int, moduloT: int
    ) -> int: ...
    def getIndex(self, *args: int) -> int:
        """Computes the rasterized index for given coordinates, optionally with modulos."""

    def getZCTCoords(self, index: int) -> List[int]:
        """Gets Z, C, T coordinates for a rasterized index."""

    def getZCTModuloCoords(self, index: int) -> List[int]:
        """Gets Z, C, T, moduloZ, moduloC, moduloT coordinates for a rasterized index."""

    def getMetadataValue(self, field: str) -> Any:
        """Retrieves a metadata field value for the current file."""

    def getSeriesMetadataValue(self, field: str) -> Any:
        """Retrieves a metadata field value for the current series."""

    def getGlobalMetadata(self) -> Dict[str, Any]:
        """Retrieves global (non-series) metadata as a dict."""

    def getSeriesMetadata(self) -> Dict[str, Any]:
        """Retrieves series-specific metadata as a dict."""

    def getCoreMetadataList(self) -> List[CoreMetadata]:
        """(Deprecated) Returns core metadata values for the current file."""

    def setMetadataFiltered(self, filter: bool) -> None:
        """Specifies whether to discard ugly metadata entries."""

    def isMetadataFiltered(self) -> bool:
        """Checks if ugly metadata filtering is enabled."""

    def setMetadataStore(self, store: MetadataStore) -> None:
        """Sets the default metadata store for this reader."""

    def getMetadataStore(self) -> MetadataStore:
        """Retrieves the current MetadataStore implementation."""

    def getMetadataStoreRoot(self) -> Any:
        """"Retrieves the fully populated root object of the current MetadataStore.""""

    def getUnderlyingReaders(self) -> Optional[List[IFormatReader]]:
        """Retrieves all underlying readers, or None if none exist."""

    def isSingleFile(self, id: str) -> bool:
        """Checks if the named file is the only file in the dataset.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If file operations fail.
        """

    def getRequiredDirectories(self, files: List[str]) -> int:
        """Determines the number of parent directories relevant to the dataset.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If file operations fail.
        """

    def getDatasetStructureDescription(self) -> str:
        """Returns a short description of the dataset structure."""

    def getPossibleDomains(self, id: str) -> List[str]:
        """Returns scientific domains in which this format is used.

        Raises:
            FormatException: If metadata parsing fails.
            IOException: If file operations fail.
        """

    def hasCompanionFiles(self) -> bool:
        """Checks if this format supports multi-file datasets."""

    def getOptimalTileWidth(self) -> int:
        """Returns the optimal sub-image width for openBytes."""

    def getOptimalTileHeight(self) -> int:
        """Returns the optimal sub-image height for openBytes."""

    def seriesToCoreIndex(self, series: int) -> int:
        """(Deprecated) Returns the core index for the specified series."""

    def coreIndexToSeries(self, index: int) -> int:
        """(Deprecated) Returns the series for the specified core index."""

    def getCoreIndex(self) -> int:
        """(Deprecated) Gets the current core index."""

    def setCoreIndex(self, no: int) -> None:
        """(Deprecated) Sets the current core index."""

    def hasFlattenedResolutions(self) -> bool:
        """Checks if resolution flattening is enabled."""

    def setFlattenedResolutions(self, flatten: bool) -> None:
        """Enables or disables resolution flattening."""

    def reopenFile(self) -> None:
        """Reopens any files closed implicitly while the reader is open.

        Raises:
            IOException: If reopening the file fails.
        """
