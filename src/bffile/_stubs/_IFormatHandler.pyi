from __future__ import annotations
from typing import List, Any, Protocol
# from java.io import Closeable, IOException
# from loci.formats.meta import IMetadataConfigurable
# from loci.formats import FormatException

class Closeable(Protocol):
    def close(self) -> None:
        """Closes the resource."""

class IFormatHandler(Closeable, IMetadataConfigurable):
    def isThisType(self, name: str) -> bool:
        """Checks if the given string is a valid filename for this file format."""
    def getFormat(self) -> str:
        """Gets the name of this file format."""
    def getSuffixes(self) -> List[str]:
        """Gets the default file suffixes for this file format."""
    def getNativeDataType(self) -> type[Any]:
        """Returns the native data type of image planes, such as byte arrays or BufferedImage."""
    def setId(self, id: str) -> None:
        """Sets the current file name.

        Raises:
            FormatException: If an error occurs configuring the format.
            IOException: If setting the ID fails due to I/O errors.
        """
