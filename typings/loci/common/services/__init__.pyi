import typing
from typing import Protocol

import java.io
import java.lang

class DependencyException(java.lang.Exception):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str, class_: type[Service]): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        class_: type[Service],
        throwable: java.lang.Throwable,
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...
    def getFailureClass(self) -> type[Service]: ...
    def toString(self) -> java.lang.String: ...

class S3ClientStat:
    def length(self) -> int: ...

class Service: ...

class ServiceException(java.lang.Exception):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...

class ServiceFactory:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    _getInstance__T = typing.TypeVar("_getInstance__T", bound=Service)  # <T>
    def getInstance(self, class_: type[_getInstance__T]) -> _getInstance__T: ...

class AbstractService(Service):
    def __init__(self): ...

class S3ClientService(Service):
    def bucketExists(self, string: java.lang.String | str) -> bool: ...
    @typing.overload
    def getObject(
        self, string: java.lang.String | str, string2: java.lang.String | str, long: int
    ) -> java.io.InputStream: ...
    @typing.overload
    def getObject(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ) -> None: ...
    def initialize(
        self,
        string: java.lang.String | str,
        int: int,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
        string4: java.lang.String | str,
        string5: java.lang.String | str,
    ) -> None: ...
    def statObject(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ) -> S3ClientStat: ...

class S3ClientServiceException(ServiceException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    @typing.overload
    def __init__(self, throwable: java.lang.Throwable): ...

class S3ClientServiceImpl(AbstractService, S3ClientService):
    def __init__(self): ...
    def bucketExists(self, string: java.lang.String | str) -> bool: ...
    @typing.overload
    def getObject(
        self, string: java.lang.String | str, string2: java.lang.String | str, long: int
    ) -> java.io.InputStream: ...
    @typing.overload
    def getObject(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ) -> None: ...
    def initialize(
        self,
        string: java.lang.String | str,
        int: int,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
        string4: java.lang.String | str,
        string5: java.lang.String | str,
    ) -> None: ...
    def statObject(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ) -> S3ClientStat: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.common.services")``.

    AbstractService: type[AbstractService]
    DependencyException: type[DependencyException]
    S3ClientService: type[S3ClientService]
    S3ClientServiceException: type[S3ClientServiceException]
    S3ClientServiceImpl: type[S3ClientServiceImpl]
    S3ClientStat: type[S3ClientStat]
    Service: type[Service]
    ServiceException: type[ServiceException]
    ServiceFactory: type[ServiceFactory]
