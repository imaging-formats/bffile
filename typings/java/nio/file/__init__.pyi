import builtins
import typing
from typing import Protocol

import java.io
import java.lang
import java.net
import java.nio.channels
import java.nio.charset
import java.nio.file.attribute
import java.nio.file.spi
import java.security
import java.util
import java.util.concurrent
import java.util.function
import java.util.stream
import jpype
import jpype.protocol

class AccessMode(java.lang.Enum["AccessMode"]):
    READ: typing.ClassVar[AccessMode] = ...
    WRITE: typing.ClassVar[AccessMode] = ...
    EXECUTE: typing.ClassVar[AccessMode] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> AccessMode: ...
    @staticmethod
    def values() -> typing.MutableSequence[AccessMode]: ...

class ClosedDirectoryStreamException(java.lang.IllegalStateException):
    def __init__(self): ...

class ClosedFileSystemException(java.lang.IllegalStateException):
    def __init__(self): ...

class ClosedWatchServiceException(java.lang.IllegalStateException):
    def __init__(self): ...

class CopyOption: ...

class DirectoryIteratorException(java.util.ConcurrentModificationException):
    def __init__(self, iOException: java.io.IOException): ...
    def getCause(self) -> java.io.IOException: ...

_DirectoryStream__Filter__T = typing.TypeVar("_DirectoryStream__Filter__T")  # <T>
_DirectoryStream__T = typing.TypeVar("_DirectoryStream__T")  # <T>

class DirectoryStream(
    java.io.Closeable,
    java.lang.Iterable[_DirectoryStream__T],
    typing.Generic[_DirectoryStream__T],
):
    def iterator(self) -> java.util.Iterator[_DirectoryStream__T]: ...
    class Filter(typing.Generic[_DirectoryStream__Filter__T]):
        def accept(self, t: _DirectoryStream__Filter__T) -> bool: ...

class FileStore:
    def getAttribute(self, string: java.lang.String | str) -> typing.Any: ...
    def getBlockSize(self) -> int: ...
    _getFileStoreAttributeView__V = typing.TypeVar(
        "_getFileStoreAttributeView__V",
        bound=java.nio.file.attribute.FileStoreAttributeView,
    )  # <V>
    def getFileStoreAttributeView(
        self, class_: builtins.type[_getFileStoreAttributeView__V]
    ) -> _getFileStoreAttributeView__V: ...
    def getTotalSpace(self) -> int: ...
    def getUnallocatedSpace(self) -> int: ...
    def getUsableSpace(self) -> int: ...
    def isReadOnly(self) -> bool: ...
    def name(self) -> java.lang.String: ...
    @typing.overload
    def supportsFileAttributeView(
        self, class_: builtins.type[java.nio.file.attribute.FileAttributeView]
    ) -> bool: ...
    @typing.overload
    def supportsFileAttributeView(self, string: java.lang.String | str) -> bool: ...
    def type(self) -> java.lang.String: ...

class FileSystem(java.io.Closeable):
    def close(self) -> None: ...
    def getFileStores(self) -> java.lang.Iterable[FileStore]: ...
    def getPath(
        self, string: java.lang.String | str, *string2: java.lang.String | str
    ) -> Path: ...
    def getPathMatcher(self, string: java.lang.String | str) -> PathMatcher: ...
    def getRootDirectories(self) -> java.lang.Iterable[Path]: ...
    def getSeparator(self) -> java.lang.String: ...
    def getUserPrincipalLookupService(
        self,
    ) -> java.nio.file.attribute.UserPrincipalLookupService: ...
    def isOpen(self) -> bool: ...
    def isReadOnly(self) -> bool: ...
    def newWatchService(self) -> WatchService: ...
    def provider(self) -> java.nio.file.spi.FileSystemProvider: ...
    def supportedFileAttributeViews(self) -> java.util.Set[java.lang.String]: ...

class FileSystemAlreadyExistsException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class FileSystemException(java.io.IOException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...
    def getFile(self) -> java.lang.String: ...
    def getMessage(self) -> java.lang.String: ...
    def getOtherFile(self) -> java.lang.String: ...
    def getReason(self) -> java.lang.String: ...

class FileSystemNotFoundException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class FileSystems:
    @staticmethod
    def getDefault() -> FileSystem: ...
    @staticmethod
    def getFileSystem(uRI: java.net.URI) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(
        uRI: java.net.URI,
        map: java.util.Map[java.lang.String | str, typing.Any]
        | typing.Mapping[java.lang.String | str, typing.Any],
    ) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(
        uRI: java.net.URI,
        map: java.util.Map[java.lang.String | str, typing.Any]
        | typing.Mapping[java.lang.String | str, typing.Any],
        classLoader: java.lang.ClassLoader,
    ) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(path: Path | jpype.protocol.SupportsPath) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(
        path: Path | jpype.protocol.SupportsPath, classLoader: java.lang.ClassLoader
    ) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(
        path: Path | jpype.protocol.SupportsPath,
        map: java.util.Map[java.lang.String | str, typing.Any]
        | typing.Mapping[java.lang.String | str, typing.Any],
    ) -> FileSystem: ...
    @typing.overload
    @staticmethod
    def newFileSystem(
        path: Path | jpype.protocol.SupportsPath,
        map: java.util.Map[java.lang.String | str, typing.Any]
        | typing.Mapping[java.lang.String | str, typing.Any],
        classLoader: java.lang.ClassLoader,
    ) -> FileSystem: ...

class FileVisitOption(java.lang.Enum["FileVisitOption"]):
    FOLLOW_LINKS: typing.ClassVar[FileVisitOption] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> FileVisitOption: ...
    @staticmethod
    def values() -> typing.MutableSequence[FileVisitOption]: ...

class FileVisitResult(java.lang.Enum["FileVisitResult"]):
    CONTINUE: typing.ClassVar[FileVisitResult] = ...
    TERMINATE: typing.ClassVar[FileVisitResult] = ...
    SKIP_SUBTREE: typing.ClassVar[FileVisitResult] = ...
    SKIP_SIBLINGS: typing.ClassVar[FileVisitResult] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> FileVisitResult: ...
    @staticmethod
    def values() -> typing.MutableSequence[FileVisitResult]: ...

_FileVisitor__T = typing.TypeVar("_FileVisitor__T")  # <T>

class FileVisitor(typing.Generic[_FileVisitor__T]):
    def postVisitDirectory(
        self, t: _FileVisitor__T, iOException: java.io.IOException
    ) -> FileVisitResult: ...
    def preVisitDirectory(
        self,
        t: _FileVisitor__T,
        basicFileAttributes: java.nio.file.attribute.BasicFileAttributes,
    ) -> FileVisitResult: ...
    def visitFile(
        self,
        t: _FileVisitor__T,
        basicFileAttributes: java.nio.file.attribute.BasicFileAttributes,
    ) -> FileVisitResult: ...
    def visitFileFailed(
        self, t: _FileVisitor__T, iOException: java.io.IOException
    ) -> FileVisitResult: ...

class Files:
    @typing.overload
    @staticmethod
    def copy(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
        *copyOption: CopyOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def copy(
        inputStream: java.io.InputStream,
        path: Path | jpype.protocol.SupportsPath,
        *copyOption: CopyOption,
    ) -> int: ...
    @typing.overload
    @staticmethod
    def copy(
        path: Path | jpype.protocol.SupportsPath, outputStream: java.io.OutputStream
    ) -> int: ...
    @staticmethod
    def createDirectories(
        path: Path | jpype.protocol.SupportsPath,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @staticmethod
    def createDirectory(
        path: Path | jpype.protocol.SupportsPath,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @staticmethod
    def createFile(
        path: Path | jpype.protocol.SupportsPath,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @staticmethod
    def createLink(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
    ) -> Path: ...
    @staticmethod
    def createSymbolicLink(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def createTempDirectory(
        string: java.lang.String | str,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def createTempDirectory(
        path: Path | jpype.protocol.SupportsPath,
        string: java.lang.String | str,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def createTempFile(
        string: java.lang.String | str,
        string2: java.lang.String | str,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def createTempFile(
        path: Path | jpype.protocol.SupportsPath,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> Path: ...
    @staticmethod
    def delete(path: Path | jpype.protocol.SupportsPath) -> None: ...
    @staticmethod
    def deleteIfExists(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @staticmethod
    def exists(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> bool: ...
    @staticmethod
    def find(
        path: Path | jpype.protocol.SupportsPath,
        int: int,
        biPredicate: java.util.function.BiPredicate[
            Path | jpype.protocol.SupportsPath,
            java.nio.file.attribute.BasicFileAttributes,
        ]
        | typing.Callable[
            [
                Path | jpype.protocol.SupportsPath,
                java.nio.file.attribute.BasicFileAttributes,
            ],
            bool,
        ],
        *fileVisitOption: FileVisitOption,
    ) -> java.util.stream.Stream[Path]: ...
    @staticmethod
    def getAttribute(
        path: Path | jpype.protocol.SupportsPath,
        string: java.lang.String | str,
        *linkOption: LinkOption,
    ) -> typing.Any: ...
    _getFileAttributeView__V = typing.TypeVar(
        "_getFileAttributeView__V", bound=java.nio.file.attribute.FileAttributeView
    )  # <V>
    @staticmethod
    def getFileAttributeView(
        path: Path | jpype.protocol.SupportsPath,
        class_: type[_getFileAttributeView__V],
        *linkOption: LinkOption,
    ) -> _getFileAttributeView__V: ...
    @staticmethod
    def getFileStore(path: Path | jpype.protocol.SupportsPath) -> FileStore: ...
    @staticmethod
    def getLastModifiedTime(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> java.nio.file.attribute.FileTime: ...
    @staticmethod
    def getOwner(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> java.nio.file.attribute.UserPrincipal: ...
    @staticmethod
    def getPosixFilePermissions(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> java.util.Set[java.nio.file.attribute.PosixFilePermission]: ...
    @staticmethod
    def isDirectory(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> bool: ...
    @staticmethod
    def isExecutable(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @staticmethod
    def isHidden(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @staticmethod
    def isReadable(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @staticmethod
    def isRegularFile(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> bool: ...
    @staticmethod
    def isSameFile(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
    ) -> bool: ...
    @staticmethod
    def isSymbolicLink(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @staticmethod
    def isWritable(path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @typing.overload
    @staticmethod
    def lines(
        path: Path | jpype.protocol.SupportsPath,
    ) -> java.util.stream.Stream[java.lang.String]: ...
    @typing.overload
    @staticmethod
    def lines(
        path: Path | jpype.protocol.SupportsPath, charset: java.nio.charset.Charset
    ) -> java.util.stream.Stream[java.lang.String]: ...
    @staticmethod
    def list(
        path: Path | jpype.protocol.SupportsPath,
    ) -> java.util.stream.Stream[Path]: ...
    @staticmethod
    def mismatch(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
    ) -> int: ...
    @staticmethod
    def move(
        path: Path | jpype.protocol.SupportsPath,
        path2: Path | jpype.protocol.SupportsPath,
        *copyOption: CopyOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def newBufferedReader(
        path: Path | jpype.protocol.SupportsPath,
    ) -> java.io.BufferedReader: ...
    @typing.overload
    @staticmethod
    def newBufferedReader(
        path: Path | jpype.protocol.SupportsPath, charset: java.nio.charset.Charset
    ) -> java.io.BufferedReader: ...
    @typing.overload
    @staticmethod
    def newBufferedWriter(
        path: Path | jpype.protocol.SupportsPath,
        charset: java.nio.charset.Charset,
        *openOption: OpenOption,
    ) -> java.io.BufferedWriter: ...
    @typing.overload
    @staticmethod
    def newBufferedWriter(
        path: Path | jpype.protocol.SupportsPath, *openOption: OpenOption
    ) -> java.io.BufferedWriter: ...
    @typing.overload
    @staticmethod
    def newByteChannel(
        path: Path | jpype.protocol.SupportsPath, *openOption: OpenOption
    ) -> java.nio.channels.SeekableByteChannel: ...
    @typing.overload
    @staticmethod
    def newByteChannel(
        path: Path | jpype.protocol.SupportsPath,
        set: java.util.Set[OpenOption],
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> java.nio.channels.SeekableByteChannel: ...
    @typing.overload
    @staticmethod
    def newDirectoryStream(
        path: Path | jpype.protocol.SupportsPath,
    ) -> DirectoryStream[Path]: ...
    @typing.overload
    @staticmethod
    def newDirectoryStream(
        path: Path | jpype.protocol.SupportsPath, string: java.lang.String | str
    ) -> DirectoryStream[Path]: ...
    @typing.overload
    @staticmethod
    def newDirectoryStream(
        path: Path | jpype.protocol.SupportsPath,
        filter: DirectoryStream.Filter[Path] | typing.Callable[[Path], bool],
    ) -> DirectoryStream[Path]: ...
    @staticmethod
    def newInputStream(
        path: Path | jpype.protocol.SupportsPath, *openOption: OpenOption
    ) -> java.io.InputStream: ...
    @staticmethod
    def newOutputStream(
        path: Path | jpype.protocol.SupportsPath, *openOption: OpenOption
    ) -> java.io.OutputStream: ...
    @staticmethod
    def notExists(
        path: Path | jpype.protocol.SupportsPath, *linkOption: LinkOption
    ) -> bool: ...
    @staticmethod
    def probeContentType(
        path: Path | jpype.protocol.SupportsPath,
    ) -> java.lang.String: ...
    @staticmethod
    def readAllBytes(
        path: Path | jpype.protocol.SupportsPath,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def readAllLines(
        path: Path | jpype.protocol.SupportsPath,
    ) -> java.util.List[java.lang.String]: ...
    @typing.overload
    @staticmethod
    def readAllLines(
        path: Path | jpype.protocol.SupportsPath, charset: java.nio.charset.Charset
    ) -> java.util.List[java.lang.String]: ...
    _readAttributes_0__A = typing.TypeVar(
        "_readAttributes_0__A", bound=java.nio.file.attribute.BasicFileAttributes
    )  # <A>
    @typing.overload
    @staticmethod
    def readAttributes(
        path: Path | jpype.protocol.SupportsPath,
        class_: type[_readAttributes_0__A],
        *linkOption: LinkOption,
    ) -> _readAttributes_0__A: ...
    @typing.overload
    @staticmethod
    def readAttributes(
        path: Path | jpype.protocol.SupportsPath,
        string: java.lang.String | str,
        *linkOption: LinkOption,
    ) -> java.util.Map[java.lang.String, typing.Any]: ...
    @typing.overload
    @staticmethod
    def readString(path: Path | jpype.protocol.SupportsPath) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def readString(
        path: Path | jpype.protocol.SupportsPath, charset: java.nio.charset.Charset
    ) -> java.lang.String: ...
    @staticmethod
    def readSymbolicLink(path: Path | jpype.protocol.SupportsPath) -> Path: ...
    @staticmethod
    def setAttribute(
        path: Path | jpype.protocol.SupportsPath,
        string: java.lang.String | str,
        object: typing.Any,
        *linkOption: LinkOption,
    ) -> Path: ...
    @staticmethod
    def setLastModifiedTime(
        path: Path | jpype.protocol.SupportsPath,
        fileTime: java.nio.file.attribute.FileTime,
    ) -> Path: ...
    @staticmethod
    def setOwner(
        path: Path | jpype.protocol.SupportsPath,
        userPrincipal: java.nio.file.attribute.UserPrincipal | typing.Callable,
    ) -> Path: ...
    @staticmethod
    def setPosixFilePermissions(
        path: Path | jpype.protocol.SupportsPath,
        set: java.util.Set[java.nio.file.attribute.PosixFilePermission],
    ) -> Path: ...
    @staticmethod
    def size(path: Path | jpype.protocol.SupportsPath) -> int: ...
    @typing.overload
    @staticmethod
    def walk(
        path: Path | jpype.protocol.SupportsPath,
        int: int,
        *fileVisitOption: FileVisitOption,
    ) -> java.util.stream.Stream[Path]: ...
    @typing.overload
    @staticmethod
    def walk(
        path: Path | jpype.protocol.SupportsPath, *fileVisitOption: FileVisitOption
    ) -> java.util.stream.Stream[Path]: ...
    @typing.overload
    @staticmethod
    def walkFileTree(
        path: Path | jpype.protocol.SupportsPath, fileVisitor: FileVisitor[Path]
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def walkFileTree(
        path: Path | jpype.protocol.SupportsPath,
        set: java.util.Set[FileVisitOption],
        int: int,
        fileVisitor: FileVisitor[Path],
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def write(
        path: Path | jpype.protocol.SupportsPath,
        byteArray: builtins.list[int] | jpype.JArray | bytes,
        *openOption: OpenOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def write(
        path: Path | jpype.protocol.SupportsPath,
        iterable: java.lang.Iterable[java.lang.CharSequence]
        | typing.Sequence[java.lang.CharSequence]
        | set[java.lang.CharSequence]
        | typing.Callable[[], java.util.Iterator[typing.Any]],
        charset: java.nio.charset.Charset,
        *openOption: OpenOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def write(
        path: Path | jpype.protocol.SupportsPath,
        iterable: java.lang.Iterable[java.lang.CharSequence]
        | typing.Sequence[java.lang.CharSequence]
        | set[java.lang.CharSequence]
        | typing.Callable[[], java.util.Iterator[typing.Any]],
        *openOption: OpenOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def writeString(
        path: Path | jpype.protocol.SupportsPath,
        charSequence: java.lang.CharSequence | str,
        charset: java.nio.charset.Charset,
        *openOption: OpenOption,
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def writeString(
        path: Path | jpype.protocol.SupportsPath,
        charSequence: java.lang.CharSequence | str,
        *openOption: OpenOption,
    ) -> Path: ...

class InvalidPathException(java.lang.IllegalArgumentException):
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, string2: java.lang.String | str, int: int
    ): ...
    def getIndex(self) -> int: ...
    def getInput(self) -> java.lang.String: ...
    def getMessage(self) -> java.lang.String: ...
    def getReason(self) -> java.lang.String: ...

class LinkPermission(java.security.BasicPermission):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ): ...

class OpenOption: ...

class PathMatcher:
    def matches(self, path: Path | jpype.protocol.SupportsPath) -> bool: ...

class Paths:
    @typing.overload
    @staticmethod
    def get(
        string: java.lang.String | str, *string2: java.lang.String | str
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def get(uRI: java.net.URI) -> Path: ...

class ProviderMismatchException(java.lang.IllegalArgumentException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class ProviderNotFoundException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class ReadOnlyFileSystemException(java.lang.UnsupportedOperationException):
    def __init__(self): ...

class StandardWatchEventKinds:
    OVERFLOW: typing.ClassVar[WatchEvent.Kind] = ...
    ENTRY_CREATE: typing.ClassVar[WatchEvent.Kind] = ...
    ENTRY_DELETE: typing.ClassVar[WatchEvent.Kind] = ...
    ENTRY_MODIFY: typing.ClassVar[WatchEvent.Kind] = ...

_WatchEvent__Kind__T = typing.TypeVar("_WatchEvent__Kind__T")  # <T>
_WatchEvent__T = typing.TypeVar("_WatchEvent__T")  # <T>

class WatchEvent(typing.Generic[_WatchEvent__T]):
    def context(self) -> _WatchEvent__T: ...
    def count(self) -> int: ...
    def kind(self) -> WatchEvent.Kind[_WatchEvent__T]: ...
    class Kind(typing.Generic[_WatchEvent__Kind__T]):
        def name(self) -> java.lang.String: ...
        def type(self) -> builtins.type[_WatchEvent__Kind__T]: ...

    class Modifier:
        def name(self) -> java.lang.String: ...

class WatchKey:
    def cancel(self) -> None: ...
    def isValid(self) -> bool: ...
    def pollEvents(self) -> java.util.List[WatchEvent[typing.Any]]: ...
    def reset(self) -> bool: ...
    def watchable(self) -> Watchable: ...

class WatchService(java.io.Closeable):
    def close(self) -> None: ...
    @typing.overload
    def poll(self) -> WatchKey: ...
    @typing.overload
    def poll(self, long: int, timeUnit: java.util.concurrent.TimeUnit) -> WatchKey: ...
    def take(self) -> WatchKey: ...

class Watchable:
    @typing.overload
    def register(
        self, watchService: WatchService, *kind: WatchEvent.Kind[typing.Any]
    ) -> WatchKey: ...
    @typing.overload
    def register(
        self,
        watchService: WatchService,
        kindArray: list[WatchEvent.Kind[typing.Any]] | jpype.JArray,
        *modifier: WatchEvent.Modifier | typing.Callable,
    ) -> WatchKey: ...

class AccessDeniedException(FileSystemException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...

class AtomicMoveNotSupportedException(FileSystemException):
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...

class DirectoryNotEmptyException(FileSystemException):
    def __init__(self, string: java.lang.String | str): ...

class FileAlreadyExistsException(FileSystemException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...

class FileSystemLoopException(FileSystemException):
    def __init__(self, string: java.lang.String | str): ...

class LinkOption(java.lang.Enum["LinkOption"], OpenOption, CopyOption):
    NOFOLLOW_LINKS: typing.ClassVar[LinkOption] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> LinkOption: ...
    @staticmethod
    def values() -> typing.MutableSequence[LinkOption]: ...

class NoSuchFileException(FileSystemException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...

class NotDirectoryException(FileSystemException):
    def __init__(self, string: java.lang.String | str): ...

class NotLinkException(FileSystemException):
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...
    @typing.overload
    def __init__(
        self,
        string: java.lang.String | str,
        string2: java.lang.String | str,
        string3: java.lang.String | str,
    ): ...

class Path(java.lang.Comparable["Path"], java.lang.Iterable["Path"], Watchable):
    def compareTo(self, path: Path | jpype.protocol.SupportsPath) -> int: ...
    @typing.overload
    def endsWith(self, path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @typing.overload
    def endsWith(self, string: java.lang.String | str) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def getFileName(self) -> Path: ...
    def getFileSystem(self) -> FileSystem: ...
    def getName(self, int: int) -> Path: ...
    def getNameCount(self) -> int: ...
    def getParent(self) -> Path: ...
    def getRoot(self) -> Path: ...
    def hashCode(self) -> int: ...
    def isAbsolute(self) -> bool: ...
    def iterator(self) -> java.util.Iterator[Path]: ...
    def normalize(self) -> Path: ...
    @typing.overload
    @staticmethod
    def of(
        string: java.lang.String | str, *string2: java.lang.String | str
    ) -> Path: ...
    @typing.overload
    @staticmethod
    def of(uRI: java.net.URI) -> Path: ...
    @typing.overload
    def register(
        self,
        watchService: WatchService,
        kindArray: list[WatchEvent.Kind[typing.Any]] | jpype.JArray,
        *modifier: WatchEvent.Modifier | typing.Callable,
    ) -> WatchKey: ...
    @typing.overload
    def register(
        self, watchService: WatchService, *kind: WatchEvent.Kind[typing.Any]
    ) -> WatchKey: ...
    def relativize(self, path: Path | jpype.protocol.SupportsPath) -> Path: ...
    @typing.overload
    def resolve(self, path: Path | jpype.protocol.SupportsPath) -> Path: ...
    @typing.overload
    def resolve(self, string: java.lang.String | str) -> Path: ...
    @typing.overload
    def resolve(
        self, string: java.lang.String | str, *string2: java.lang.String | str
    ) -> Path: ...
    @typing.overload
    def resolve(
        self,
        path: Path | jpype.protocol.SupportsPath,
        *path2: Path | jpype.protocol.SupportsPath,
    ) -> Path: ...
    @typing.overload
    def resolveSibling(self, string: java.lang.String | str) -> Path: ...
    @typing.overload
    def resolveSibling(self, path: Path | jpype.protocol.SupportsPath) -> Path: ...
    @typing.overload
    def startsWith(self, path: Path | jpype.protocol.SupportsPath) -> bool: ...
    @typing.overload
    def startsWith(self, string: java.lang.String | str) -> bool: ...
    def subpath(self, int: int, int2: int) -> Path: ...
    def toAbsolutePath(self) -> Path: ...
    def toFile(self) -> java.io.File: ...
    def toRealPath(self, *linkOption: LinkOption) -> Path: ...
    def toString(self) -> java.lang.String: ...
    def toUri(self) -> java.net.URI: ...

_SecureDirectoryStream__T = typing.TypeVar("_SecureDirectoryStream__T")  # <T>

class SecureDirectoryStream(
    DirectoryStream[_SecureDirectoryStream__T],
    typing.Generic[_SecureDirectoryStream__T],
):
    def deleteDirectory(self, t: _SecureDirectoryStream__T) -> None: ...
    def deleteFile(self, t: _SecureDirectoryStream__T) -> None: ...
    _getFileAttributeView_0__V = typing.TypeVar(
        "_getFileAttributeView_0__V", bound=java.nio.file.attribute.FileAttributeView
    )  # <V>
    _getFileAttributeView_1__V = typing.TypeVar(
        "_getFileAttributeView_1__V", bound=java.nio.file.attribute.FileAttributeView
    )  # <V>
    @typing.overload
    def getFileAttributeView(
        self, class_: type[_getFileAttributeView_0__V]
    ) -> _getFileAttributeView_0__V: ...
    @typing.overload
    def getFileAttributeView(
        self,
        t: _SecureDirectoryStream__T,
        class_: type[_getFileAttributeView_1__V],
        *linkOption: LinkOption,
    ) -> _getFileAttributeView_1__V: ...
    def move(
        self,
        t: _SecureDirectoryStream__T,
        secureDirectoryStream: SecureDirectoryStream[_SecureDirectoryStream__T],
        t2: _SecureDirectoryStream__T,
    ) -> None: ...
    def newByteChannel(
        self,
        t: _SecureDirectoryStream__T,
        set: java.util.Set[OpenOption],
        *fileAttribute: java.nio.file.attribute.FileAttribute[typing.Any],
    ) -> java.nio.channels.SeekableByteChannel: ...
    def newDirectoryStream(
        self, t: _SecureDirectoryStream__T, *linkOption: LinkOption
    ) -> SecureDirectoryStream[_SecureDirectoryStream__T]: ...

_SimpleFileVisitor__T = typing.TypeVar("_SimpleFileVisitor__T")  # <T>

class SimpleFileVisitor(
    FileVisitor[_SimpleFileVisitor__T], typing.Generic[_SimpleFileVisitor__T]
):
    def postVisitDirectory(
        self, t: _SimpleFileVisitor__T, iOException: java.io.IOException
    ) -> FileVisitResult: ...
    def preVisitDirectory(
        self,
        t: _SimpleFileVisitor__T,
        basicFileAttributes: java.nio.file.attribute.BasicFileAttributes,
    ) -> FileVisitResult: ...
    def visitFile(
        self,
        t: _SimpleFileVisitor__T,
        basicFileAttributes: java.nio.file.attribute.BasicFileAttributes,
    ) -> FileVisitResult: ...
    def visitFileFailed(
        self, t: _SimpleFileVisitor__T, iOException: java.io.IOException
    ) -> FileVisitResult: ...

class StandardCopyOption(java.lang.Enum["StandardCopyOption"], CopyOption):
    REPLACE_EXISTING: typing.ClassVar[StandardCopyOption] = ...
    COPY_ATTRIBUTES: typing.ClassVar[StandardCopyOption] = ...
    ATOMIC_MOVE: typing.ClassVar[StandardCopyOption] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> StandardCopyOption: ...
    @staticmethod
    def values() -> typing.MutableSequence[StandardCopyOption]: ...

class StandardOpenOption(java.lang.Enum["StandardOpenOption"], OpenOption):
    READ: typing.ClassVar[StandardOpenOption] = ...
    WRITE: typing.ClassVar[StandardOpenOption] = ...
    APPEND: typing.ClassVar[StandardOpenOption] = ...
    TRUNCATE_EXISTING: typing.ClassVar[StandardOpenOption] = ...
    CREATE: typing.ClassVar[StandardOpenOption] = ...
    CREATE_NEW: typing.ClassVar[StandardOpenOption] = ...
    DELETE_ON_CLOSE: typing.ClassVar[StandardOpenOption] = ...
    SPARSE: typing.ClassVar[StandardOpenOption] = ...
    SYNC: typing.ClassVar[StandardOpenOption] = ...
    DSYNC: typing.ClassVar[StandardOpenOption] = ...
    _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
    @typing.overload
    @staticmethod
    def valueOf(
        class_: type[_valueOf_0__T], string: java.lang.String | str
    ) -> _valueOf_0__T: ...
    @typing.overload
    @staticmethod
    def valueOf(string: java.lang.String | str) -> StandardOpenOption: ...
    @staticmethod
    def values() -> typing.MutableSequence[StandardOpenOption]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.nio.file")``.

    AccessDeniedException: type[AccessDeniedException]
    AccessMode: type[AccessMode]
    AtomicMoveNotSupportedException: type[AtomicMoveNotSupportedException]
    ClosedDirectoryStreamException: type[ClosedDirectoryStreamException]
    ClosedFileSystemException: type[ClosedFileSystemException]
    ClosedWatchServiceException: type[ClosedWatchServiceException]
    CopyOption: type[CopyOption]
    DirectoryIteratorException: type[DirectoryIteratorException]
    DirectoryNotEmptyException: type[DirectoryNotEmptyException]
    DirectoryStream: type[DirectoryStream]
    FileAlreadyExistsException: type[FileAlreadyExistsException]
    FileStore: type[FileStore]
    FileSystem: type[FileSystem]
    FileSystemAlreadyExistsException: type[FileSystemAlreadyExistsException]
    FileSystemException: type[FileSystemException]
    FileSystemLoopException: type[FileSystemLoopException]
    FileSystemNotFoundException: type[FileSystemNotFoundException]
    FileSystems: type[FileSystems]
    FileVisitOption: type[FileVisitOption]
    FileVisitResult: type[FileVisitResult]
    FileVisitor: type[FileVisitor]
    Files: type[Files]
    InvalidPathException: type[InvalidPathException]
    LinkOption: type[LinkOption]
    LinkPermission: type[LinkPermission]
    NoSuchFileException: type[NoSuchFileException]
    NotDirectoryException: type[NotDirectoryException]
    NotLinkException: type[NotLinkException]
    OpenOption: type[OpenOption]
    Path: type[Path]
    PathMatcher: type[PathMatcher]
    Paths: type[Paths]
    ProviderMismatchException: type[ProviderMismatchException]
    ProviderNotFoundException: type[ProviderNotFoundException]
    ReadOnlyFileSystemException: type[ReadOnlyFileSystemException]
    SecureDirectoryStream: type[SecureDirectoryStream]
    SimpleFileVisitor: type[SimpleFileVisitor]
    StandardCopyOption: type[StandardCopyOption]
    StandardOpenOption: type[StandardOpenOption]
    StandardWatchEventKinds: type[StandardWatchEventKinds]
    WatchEvent: type[WatchEvent]
    WatchKey: type[WatchKey]
    WatchService: type[WatchService]
    Watchable: type[Watchable]
    attribute: java.nio.file.attribute.__module_protocol__
    spi: java.nio.file.spi.__module_protocol__
