from __future__ import annotations
from typing import Container, Set
# from loci.formats.in import MetadataLevel, MetadataOptions


class IMetadataConfigurable:
    def getSupportedMetadataLevels(self) -> Container[MetadataLevel]:
        """Returns the set of supported metadata levels."""

    def setMetadataOptions(self, options: MetadataOptions) -> None:
        """Configures the reader with the given metadata options."""

    def getMetadataOptions(self) -> MetadataOptions:
        """Retrieves the current metadata options."""
