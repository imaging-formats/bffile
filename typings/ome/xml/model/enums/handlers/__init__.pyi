import typing
from typing import Protocol

import java.lang
import ome.units.quantity
import ome.units.unit
import ome.xml.model.enums
import ome.xml.model.primitives

class IEnumerationHandler:
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class AcquisitionModeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class ArcTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class BinningEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class CompressionEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class ContrastMethodEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class CorrectionEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class DetectorTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class DimensionOrderEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class ExperimentTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class FilamentTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class FillRuleEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class FilterTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class FontFamilyEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class FontStyleEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class IlluminationTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class ImmersionEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class LaserMediumEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class LaserTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class MarkerEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class MediumEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class MicrobeamManipulationTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class MicroscopeTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class NamingConventionEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class PixelTypeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class PulseEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...

class UnitsElectricPotentialEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsElectricPotential: ome.xml.model.enums.UnitsElectricPotential,
    ) -> ome.units.unit.Unit[ome.units.quantity.ElectricPotential]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, electricPotential: ome.units.quantity.ElectricPotential
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T,
        unitsElectricPotential: ome.xml.model.enums.UnitsElectricPotential,
    ) -> ome.units.quantity.ElectricPotential: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T,
        unitsElectricPotential: ome.xml.model.enums.UnitsElectricPotential,
    ) -> ome.units.quantity.ElectricPotential: ...

class UnitsFrequencyEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsFrequency: ome.xml.model.enums.UnitsFrequency,
    ) -> ome.units.unit.Unit[ome.units.quantity.Frequency]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, frequency: ome.units.quantity.Frequency
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsFrequency: ome.xml.model.enums.UnitsFrequency
    ) -> ome.units.quantity.Frequency: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsFrequency: ome.xml.model.enums.UnitsFrequency
    ) -> ome.units.quantity.Frequency: ...

class UnitsLengthEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsLength: ome.xml.model.enums.UnitsLength,
    ) -> ome.units.unit.Unit[ome.units.quantity.Length]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, length: ome.units.quantity.Length
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsLength: ome.xml.model.enums.UnitsLength
    ) -> ome.units.quantity.Length: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsLength: ome.xml.model.enums.UnitsLength
    ) -> ome.units.quantity.Length: ...

class UnitsPowerEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsPower: ome.xml.model.enums.UnitsPower,
    ) -> ome.units.unit.Unit[ome.units.quantity.Power]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, power: ome.units.quantity.Power
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsPower: ome.xml.model.enums.UnitsPower
    ) -> ome.units.quantity.Power: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsPower: ome.xml.model.enums.UnitsPower
    ) -> ome.units.quantity.Power: ...

class UnitsPressureEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsPressure: ome.xml.model.enums.UnitsPressure,
    ) -> ome.units.unit.Unit[ome.units.quantity.Pressure]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, pressure: ome.units.quantity.Pressure
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsPressure: ome.xml.model.enums.UnitsPressure
    ) -> ome.units.quantity.Pressure: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsPressure: ome.xml.model.enums.UnitsPressure
    ) -> ome.units.quantity.Pressure: ...

class UnitsTemperatureEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsTemperature: ome.xml.model.enums.UnitsTemperature,
    ) -> ome.units.unit.Unit[ome.units.quantity.Temperature]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, temperature: ome.units.quantity.Temperature
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsTemperature: ome.xml.model.enums.UnitsTemperature
    ) -> ome.units.quantity.Temperature: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsTemperature: ome.xml.model.enums.UnitsTemperature
    ) -> ome.units.quantity.Temperature: ...

class UnitsTimeEnumHandler(IEnumerationHandler):
    def __init__(self): ...
    @staticmethod
    def getBaseUnit(
        unitsTime: ome.xml.model.enums.UnitsTime,
    ) -> ome.units.unit.Unit[ome.units.quantity.Time]: ...
    def getEntity(self) -> type[ome.xml.model.enums.Enumeration]: ...
    @typing.overload
    def getEnumeration(
        self, string: java.lang.String | str
    ) -> ome.xml.model.enums.Enumeration: ...
    @typing.overload
    def getEnumeration(
        self, time: ome.units.quantity.Time
    ) -> ome.xml.model.enums.Enumeration: ...
    _getQuantity_0__T = typing.TypeVar(
        "_getQuantity_0__T", bound=java.lang.Number
    )  # <T>
    _getQuantity_1__T = typing.TypeVar(
        "_getQuantity_1__T", bound=ome.xml.model.primitives.PrimitiveNumber
    )  # <T>
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_0__T, unitsTime: ome.xml.model.enums.UnitsTime
    ) -> ome.units.quantity.Time: ...
    @typing.overload
    @staticmethod
    def getQuantity(
        t: _getQuantity_1__T, unitsTime: ome.xml.model.enums.UnitsTime
    ) -> ome.units.quantity.Time: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.xml.model.enums.handlers")``.

    AcquisitionModeEnumHandler: type[AcquisitionModeEnumHandler]
    ArcTypeEnumHandler: type[ArcTypeEnumHandler]
    BinningEnumHandler: type[BinningEnumHandler]
    CompressionEnumHandler: type[CompressionEnumHandler]
    ContrastMethodEnumHandler: type[ContrastMethodEnumHandler]
    CorrectionEnumHandler: type[CorrectionEnumHandler]
    DetectorTypeEnumHandler: type[DetectorTypeEnumHandler]
    DimensionOrderEnumHandler: type[DimensionOrderEnumHandler]
    ExperimentTypeEnumHandler: type[ExperimentTypeEnumHandler]
    FilamentTypeEnumHandler: type[FilamentTypeEnumHandler]
    FillRuleEnumHandler: type[FillRuleEnumHandler]
    FilterTypeEnumHandler: type[FilterTypeEnumHandler]
    FontFamilyEnumHandler: type[FontFamilyEnumHandler]
    FontStyleEnumHandler: type[FontStyleEnumHandler]
    IEnumerationHandler: type[IEnumerationHandler]
    IlluminationTypeEnumHandler: type[IlluminationTypeEnumHandler]
    ImmersionEnumHandler: type[ImmersionEnumHandler]
    LaserMediumEnumHandler: type[LaserMediumEnumHandler]
    LaserTypeEnumHandler: type[LaserTypeEnumHandler]
    MarkerEnumHandler: type[MarkerEnumHandler]
    MediumEnumHandler: type[MediumEnumHandler]
    MicrobeamManipulationTypeEnumHandler: type[MicrobeamManipulationTypeEnumHandler]
    MicroscopeTypeEnumHandler: type[MicroscopeTypeEnumHandler]
    NamingConventionEnumHandler: type[NamingConventionEnumHandler]
    PixelTypeEnumHandler: type[PixelTypeEnumHandler]
    PulseEnumHandler: type[PulseEnumHandler]
    UnitsElectricPotentialEnumHandler: type[UnitsElectricPotentialEnumHandler]
    UnitsFrequencyEnumHandler: type[UnitsFrequencyEnumHandler]
    UnitsLengthEnumHandler: type[UnitsLengthEnumHandler]
    UnitsPowerEnumHandler: type[UnitsPowerEnumHandler]
    UnitsPressureEnumHandler: type[UnitsPressureEnumHandler]
    UnitsTemperatureEnumHandler: type[UnitsTemperatureEnumHandler]
    UnitsTimeEnumHandler: type[UnitsTimeEnumHandler]
