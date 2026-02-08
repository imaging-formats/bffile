import typing
from typing import Protocol

class POIFSConstants:
    END_OF_CHAIN: typing.ClassVar[int] = ...
    PROPERTY_SIZE: typing.ClassVar[int] = ...
    UNUSED_BLOCK: typing.ClassVar[int] = ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.poifs.common")``.

    POIFSConstants: type[POIFSConstants]
