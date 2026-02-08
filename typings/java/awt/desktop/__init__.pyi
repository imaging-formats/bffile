import typing
from typing import Protocol

import java.io
import java.lang
import java.net
import java.util
import jpype.protocol

class AboutHandler:
    def handleAbout(self, aboutEvent: AboutEvent) -> None: ...

class AppEvent(java.util.EventObject): ...

class OpenFilesHandler:
    def openFiles(self, openFilesEvent: OpenFilesEvent) -> None: ...

class OpenURIHandler:
    def openURI(self, openURIEvent: OpenURIEvent) -> None: ...

class PreferencesHandler:
    def handlePreferences(self, preferencesEvent: PreferencesEvent) -> None: ...

class PrintFilesHandler:
    def printFiles(self, printFilesEvent: PrintFilesEvent) -> None: ...

class QuitHandler:
    def handleQuitRequestWith(
        self, quitEvent: QuitEvent, quitResponse: QuitResponse
    ) -> None: ...

class QuitResponse:
    def cancelQuit(self) -> None: ...
    def performQuit(self) -> None: ...

class QuitStrategy(java.lang.Enum["QuitStrategy"]):
    NORMAL_EXIT: typing.ClassVar[QuitStrategy] = ...
    CLOSE_ALL_WINDOWS: typing.ClassVar[QuitStrategy] = ...
    _valueOf_1__T = typing.TypeVar("_valueOf_1__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> QuitStrategy: ...
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_1__T], string: java.lang.String | str
    ) -> _valueOf_1__T: ...
    @staticmethod
    def values() -> typing.MutableSequence[QuitStrategy]: ...

class SystemEventListener(java.util.EventListener): ...

class AboutEvent(AppEvent):
    def __init__(self): ...

class AppForegroundEvent(AppEvent):
    def __init__(self): ...

class AppForegroundListener(SystemEventListener):
    def appMovedToBackground(self, appForegroundEvent: AppForegroundEvent) -> None: ...
    def appRaisedToForeground(self, appForegroundEvent: AppForegroundEvent) -> None: ...

class AppHiddenEvent(AppEvent):
    def __init__(self): ...

class AppHiddenListener(SystemEventListener):
    def appHidden(self, appHiddenEvent: AppHiddenEvent) -> None: ...
    def appUnhidden(self, appHiddenEvent: AppHiddenEvent) -> None: ...

class AppReopenedEvent(AppEvent):
    def __init__(self): ...

class AppReopenedListener(SystemEventListener):
    def appReopened(self, appReopenedEvent: AppReopenedEvent) -> None: ...

class FilesEvent(AppEvent):
    def getFiles(self) -> java.util.List[java.io.File]: ...

class OpenURIEvent(AppEvent):
    def __init__(self, uRI: java.net.URI): ...
    def getURI(self) -> java.net.URI: ...

class PreferencesEvent(AppEvent):
    def __init__(self): ...

class QuitEvent(AppEvent):
    def __init__(self): ...

class ScreenSleepEvent(AppEvent):
    def __init__(self): ...

class ScreenSleepListener(SystemEventListener):
    def screenAboutToSleep(self, screenSleepEvent: ScreenSleepEvent) -> None: ...
    def screenAwoke(self, screenSleepEvent: ScreenSleepEvent) -> None: ...

class SystemSleepEvent(AppEvent):
    def __init__(self): ...

class SystemSleepListener(SystemEventListener):
    def systemAboutToSleep(self, systemSleepEvent: SystemSleepEvent) -> None: ...
    def systemAwoke(self, systemSleepEvent: SystemSleepEvent) -> None: ...

class UserSessionEvent(AppEvent):
    def __init__(self, reason: UserSessionEvent.Reason): ...
    def getReason(self) -> UserSessionEvent.Reason: ...
    class Reason(java.lang.Enum["UserSessionEvent.Reason"]):
        UNSPECIFIED: typing.ClassVar[UserSessionEvent.Reason] = ...
        CONSOLE: typing.ClassVar[UserSessionEvent.Reason] = ...
        REMOTE: typing.ClassVar[UserSessionEvent.Reason] = ...
        LOCK: typing.ClassVar[UserSessionEvent.Reason] = ...
        _valueOf_1__T = typing.TypeVar("_valueOf_1__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(string: java.lang.String | str) -> UserSessionEvent.Reason: ...
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_1__T], string: java.lang.String | str
        ) -> _valueOf_1__T: ...
        @staticmethod
        def values() -> typing.MutableSequence[UserSessionEvent.Reason]: ...

class UserSessionListener(SystemEventListener):
    def userSessionActivated(self, userSessionEvent: UserSessionEvent) -> None: ...
    def userSessionDeactivated(self, userSessionEvent: UserSessionEvent) -> None: ...

class OpenFilesEvent(FilesEvent):
    def __init__(
        self,
        list: java.util.List[java.io.File | jpype.protocol.SupportsPath],
        string: java.lang.String | str,
    ): ...
    def getSearchTerm(self) -> java.lang.String: ...

class PrintFilesEvent(FilesEvent):
    def __init__(
        self, list: java.util.List[java.io.File | jpype.protocol.SupportsPath]
    ): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.awt.desktop")``.

    AboutEvent: type[AboutEvent]
    AboutHandler: type[AboutHandler]
    AppEvent: type[AppEvent]
    AppForegroundEvent: type[AppForegroundEvent]
    AppForegroundListener: type[AppForegroundListener]
    AppHiddenEvent: type[AppHiddenEvent]
    AppHiddenListener: type[AppHiddenListener]
    AppReopenedEvent: type[AppReopenedEvent]
    AppReopenedListener: type[AppReopenedListener]
    FilesEvent: type[FilesEvent]
    OpenFilesEvent: type[OpenFilesEvent]
    OpenFilesHandler: type[OpenFilesHandler]
    OpenURIEvent: type[OpenURIEvent]
    OpenURIHandler: type[OpenURIHandler]
    PreferencesEvent: type[PreferencesEvent]
    PreferencesHandler: type[PreferencesHandler]
    PrintFilesEvent: type[PrintFilesEvent]
    PrintFilesHandler: type[PrintFilesHandler]
    QuitEvent: type[QuitEvent]
    QuitHandler: type[QuitHandler]
    QuitResponse: type[QuitResponse]
    QuitStrategy: type[QuitStrategy]
    ScreenSleepEvent: type[ScreenSleepEvent]
    ScreenSleepListener: type[ScreenSleepListener]
    SystemEventListener: type[SystemEventListener]
    SystemSleepEvent: type[SystemSleepEvent]
    SystemSleepListener: type[SystemSleepListener]
    UserSessionEvent: type[UserSessionEvent]
    UserSessionListener: type[UserSessionListener]
