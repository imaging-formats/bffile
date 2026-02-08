"""Type-safe java class imports, using our local ./typings stubs for type hints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import jpype
import scyjava

from ._java_stuff import start_jvm

if TYPE_CHECKING:
    from loci.common.services import ServiceFactory
    from loci.formats import FormatTools, ImageReader, Memoizer, Modulo
    from loci.formats.in_ import DynamicMetadataOptions
    from loci.formats.ome import OMEPyramidStore
    from loci.formats.services import OMEXMLService
    from ome.units.quantity import Quantity as OMEQuantity
    from pint.facets.plain import PlainQuantity


@jpype.JImplementationFor("java.util.Map")
class _Map(Mapping):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"


@jpype.JImplementationFor("java.util.List")
class _ArrayList(Sequence):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"


@jpype.JImplementationFor("java.util.Set")
class _Set(Set):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({set(self)})"


@jpype.JImplementationFor("ome.units.quantity.Quantity")
class Quantity:
    def __repr__(self) -> str:
        obj = cast("OMEQuantity", self)
        return f"{obj.value()} {obj.unit().getSymbol()}"

    def to_pint(self) -> PlainQuantity:
        """Convert to pint Quantity."""
        from ome_types.units import ureg

        obj = cast("OMEQuantity", self)
        symbol = str(obj.unit().getSymbol()).replace(" ", "_")
        return ureg.Quantity(obj.value(), symbol)


@jpype.JImplementationFor("loci.formats.Modulo")
class _Modulo:
    def __repr__(self) -> str:
        obj = cast("Modulo", self)
        fields = []
        for name in ["start", "end", "step", "labels", "type", "unit"]:
            if (attr := getattr(obj, name)) is not None:
                fields.append(f"{name}={attr!r}")
        return f"Modulo(parentDimension={obj.parentDimension!r}, {', '.join(fields)})"


@jpype.JImplementationFor("loci.formats.CoreMetadata")
class _CoreMetadata:
    def __repr__(self) -> str:
        return f"\n{self}\n"


# ------------------------ jimport casts ------------------------------


@overload
def jimport(
    classname: Literal["ome.units.quantity.Quantity"],
) -> type[Quantity]: ...
@overload
def jimport(
    classname: Literal["loci.formats.ome.OMEPyramidStore"],
) -> type[OMEPyramidStore]: ...
@overload
def jimport(
    classname: Literal["loci.formats.in_.DynamicMetadataOptions"],
) -> type[DynamicMetadataOptions]: ...
@overload
def jimport(
    classname: Literal["loci.formats.FormatTools"],
) -> type[FormatTools]: ...
@overload
def jimport(
    classname: Literal["loci.formats.services.OMEXMLService"],
) -> type[OMEXMLService]: ...
@overload
def jimport(
    classname: Literal["loci.common.services.ServiceFactory"],
) -> type[ServiceFactory]: ...
@overload
def jimport(classname: Literal["loci.formats.ImageReader"]) -> type[ImageReader]: ...
@overload
def jimport(classname: Literal["loci.formats.Memoizer"]) -> type[Memoizer]: ...
@overload
def jimport(classname: str) -> Any: ...
def jimport(classname: str) -> Any:
    start_jvm()
    return scyjava.jimport(classname)
