import typing
from typing import Protocol

import java.io
import java.lang
import java.util

class LocaleServiceProvider:
    def getAvailableLocales(self) -> typing.MutableSequence[java.util.Locale]: ...
    def isSupportedLocale(self, locale: java.util.Locale) -> bool: ...

class ResourceBundleControlProvider:
    def getControl(
        self, string: java.lang.String | str
    ) -> java.util.ResourceBundle.Control: ...

class ResourceBundleProvider:
    def getBundle(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.util.ResourceBundle: ...

class ToolProvider:
    def description(self) -> java.util.Optional[java.lang.String]: ...
    @staticmethod
    def findFirst(
        string: java.lang.String | str,
    ) -> java.util.Optional[ToolProvider]: ...
    def name(self) -> java.lang.String: ...
    @typing.overload
    def run(
        self,
        printWriter: java.io.PrintWriter,
        printWriter2: java.io.PrintWriter,
        *string: java.lang.String | str,
    ) -> int: ...
    @typing.overload
    def run(
        self,
        printStream: java.io.PrintStream,
        printStream2: java.io.PrintStream,
        *string: java.lang.String | str,
    ) -> int: ...

class AbstractResourceBundleProvider(ResourceBundleProvider):
    def getBundle(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.util.ResourceBundle: ...

class CalendarDataProvider(LocaleServiceProvider):
    def getFirstDayOfWeek(self, locale: java.util.Locale) -> int: ...
    def getMinimalDaysInFirstWeek(self, locale: java.util.Locale) -> int: ...

class CalendarNameProvider(LocaleServiceProvider):
    def getDisplayName(
        self,
        string: java.lang.String | str,
        int: int,
        int2: int,
        int3: int,
        locale: java.util.Locale,
    ) -> java.lang.String: ...
    def getDisplayNames(
        self,
        string: java.lang.String | str,
        int: int,
        int2: int,
        locale: java.util.Locale,
    ) -> java.util.Map[java.lang.String, int]: ...

class CurrencyNameProvider(LocaleServiceProvider):
    def getDisplayName(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...
    def getSymbol(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...

class LocaleNameProvider(LocaleServiceProvider):
    def getDisplayCountry(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...
    def getDisplayLanguage(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...
    def getDisplayScript(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...
    def getDisplayUnicodeExtensionKey(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...
    def getDisplayUnicodeExtensionType(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        locale: java.util.Locale,
    ) -> java.lang.String: ...
    def getDisplayVariant(
        self, string: java.lang.String | str, locale: java.util.Locale
    ) -> java.lang.String: ...

class TimeZoneNameProvider(LocaleServiceProvider):
    def getDisplayName(
        self,
        string: java.lang.String | str,
        boolean: bool,
        int: int,
        locale: java.util.Locale,
    ) -> java.lang.String: ...
    def getGenericDisplayName(
        self, string: java.lang.String | str, int: int, locale: java.util.Locale
    ) -> java.lang.String: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.util.spi")``.

    AbstractResourceBundleProvider: type[AbstractResourceBundleProvider]
    CalendarDataProvider: type[CalendarDataProvider]
    CalendarNameProvider: type[CalendarNameProvider]
    CurrencyNameProvider: type[CurrencyNameProvider]
    LocaleNameProvider: type[LocaleNameProvider]
    LocaleServiceProvider: type[LocaleServiceProvider]
    ResourceBundleControlProvider: type[ResourceBundleControlProvider]
    ResourceBundleProvider: type[ResourceBundleProvider]
    TimeZoneNameProvider: type[TimeZoneNameProvider]
    ToolProvider: type[ToolProvider]
