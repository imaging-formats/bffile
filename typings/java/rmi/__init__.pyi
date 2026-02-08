import typing
from typing import Protocol

import java.io
import java.lang
import java.rmi.dgc
import java.rmi.registry
import java.rmi.server

class AlreadyBoundException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

_MarshalledObject__T = typing.TypeVar("_MarshalledObject__T")  # <T>

class MarshalledObject(java.io.Serializable, typing.Generic[_MarshalledObject__T]):
    def __init__(self, t: _MarshalledObject__T): ...
    def equals(self, object: typing.Any) -> bool: ...
    def get(self) -> _MarshalledObject__T: ...
    def hashCode(self) -> int: ...

class Naming:
    @staticmethod
    def bind(string: java.lang.String | str, remote: Remote) -> None: ...
    @staticmethod
    def list(
        string: java.lang.String | str,
    ) -> typing.MutableSequence[java.lang.String]: ...
    @staticmethod
    def lookup(string: java.lang.String | str) -> Remote: ...
    @staticmethod
    def rebind(string: java.lang.String | str, remote: Remote) -> None: ...
    @staticmethod
    def unbind(string: java.lang.String | str) -> None: ...

class NotBoundException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class RMISecurityException(java.lang.SecurityException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ): ...

class RMISecurityManager(java.lang.SecurityManager):
    def __init__(self): ...

class Remote: ...

class RemoteException(java.io.IOException):
    detail: java.lang.Throwable = ...
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, throwable: java.lang.Throwable
    ): ...
    def getCause(self) -> java.lang.Throwable: ...
    def getMessage(self) -> java.lang.String: ...

class AccessException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class ConnectException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class ConnectIOException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class MarshalException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class NoSuchObjectException(RemoteException):
    def __init__(self, string: java.lang.String | str): ...

class ServerError(RemoteException):
    def __init__(self, string: java.lang.String | str, error: java.lang.Error): ...

class ServerException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class ServerRuntimeException(RemoteException):
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class StubNotFoundException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class UnexpectedException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class UnknownHostException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class UnmarshalException(RemoteException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, exception: java.lang.Exception
    ): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.rmi")``.

    AccessException: type[AccessException]
    AlreadyBoundException: type[AlreadyBoundException]
    ConnectException: type[ConnectException]
    ConnectIOException: type[ConnectIOException]
    MarshalException: type[MarshalException]
    MarshalledObject: type[MarshalledObject]
    Naming: type[Naming]
    NoSuchObjectException: type[NoSuchObjectException]
    NotBoundException: type[NotBoundException]
    RMISecurityException: type[RMISecurityException]
    RMISecurityManager: type[RMISecurityManager]
    Remote: type[Remote]
    RemoteException: type[RemoteException]
    ServerError: type[ServerError]
    ServerException: type[ServerException]
    ServerRuntimeException: type[ServerRuntimeException]
    StubNotFoundException: type[StubNotFoundException]
    UnexpectedException: type[UnexpectedException]
    UnknownHostException: type[UnknownHostException]
    UnmarshalException: type[UnmarshalException]
    dgc: java.rmi.dgc.__module_protocol__
    registry: java.rmi.registry.__module_protocol__
    server: java.rmi.server.__module_protocol__
