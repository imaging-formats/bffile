import typing
from typing import Protocol

import java.io
import java.lang
import java.net
import java.nio
import java.nio.charset
import java.nio.file
import java.time
import java.util
import java.util.concurrent
import java.util.function
import java.util.stream
import javax.net.ssl
import jpype
import jpype.protocol

class HttpClient(java.lang.AutoCloseable):
    def authenticator(self) -> java.util.Optional[java.net.Authenticator]: ...
    def awaitTermination(self, duration: java.time.Duration) -> bool: ...
    def close(self) -> None: ...
    def connectTimeout(self) -> java.util.Optional[java.time.Duration]: ...
    def cookieHandler(self) -> java.util.Optional[java.net.CookieHandler]: ...
    def executor(self) -> java.util.Optional[java.util.concurrent.Executor]: ...
    def followRedirects(self) -> HttpClient.Redirect: ...
    def isTerminated(self) -> bool: ...
    @staticmethod
    def newBuilder() -> HttpClient.Builder: ...
    @staticmethod
    def newHttpClient() -> HttpClient: ...
    def newWebSocketBuilder(self) -> WebSocket.Builder: ...
    def proxy(self) -> java.util.Optional[java.net.ProxySelector]: ...
    _send__T = typing.TypeVar("_send__T")  # <T>
    def send(
        self,
        httpRequest: HttpRequest,
        bodyHandler: HttpResponse.BodyHandler[_send__T]
        | typing.Callable[
            [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
        ],
    ) -> HttpResponse[_send__T]: ...
    _sendAsync_0__T = typing.TypeVar("_sendAsync_0__T")  # <T>
    _sendAsync_1__T = typing.TypeVar("_sendAsync_1__T")  # <T>
    @typing.overload
    def sendAsync(
        self,
        httpRequest: HttpRequest,
        bodyHandler: HttpResponse.BodyHandler[_sendAsync_0__T]
        | typing.Callable[
            [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
        ],
    ) -> java.util.concurrent.CompletableFuture[HttpResponse[_sendAsync_0__T]]: ...
    @typing.overload
    def sendAsync(
        self,
        httpRequest: HttpRequest,
        bodyHandler: HttpResponse.BodyHandler[_sendAsync_1__T]
        | typing.Callable[
            [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
        ],
        pushPromiseHandler: HttpResponse.PushPromiseHandler[_sendAsync_1__T]
        | typing.Callable[
            [
                HttpRequest,
                HttpRequest,
                java.util.function.Function[
                    HttpResponse.BodyHandler[typing.Any],
                    java.util.concurrent.CompletableFuture[HttpResponse[typing.Any]],
                ],
            ],
            None,
        ],
    ) -> java.util.concurrent.CompletableFuture[HttpResponse[_sendAsync_1__T]]: ...
    def shutdown(self) -> None: ...
    def shutdownNow(self) -> None: ...
    def sslContext(self) -> javax.net.ssl.SSLContext: ...
    def sslParameters(self) -> javax.net.ssl.SSLParameters: ...
    def version(self) -> HttpClient.Version: ...
    class Builder:
        NO_PROXY: typing.ClassVar[java.net.ProxySelector] = ...
        def authenticator(
            self, authenticator: java.net.Authenticator
        ) -> HttpClient.Builder: ...
        def build(self) -> HttpClient: ...
        def connectTimeout(
            self, duration: java.time.Duration
        ) -> HttpClient.Builder: ...
        def cookieHandler(
            self, cookieHandler: java.net.CookieHandler
        ) -> HttpClient.Builder: ...
        def executor(
            self, executor: java.util.concurrent.Executor | typing.Callable
        ) -> HttpClient.Builder: ...
        def followRedirects(
            self, redirect: HttpClient.Redirect
        ) -> HttpClient.Builder: ...
        def localAddress(
            self, inetAddress: java.net.InetAddress
        ) -> HttpClient.Builder: ...
        def priority(self, int: int) -> HttpClient.Builder: ...
        def proxy(
            self, proxySelector: java.net.ProxySelector
        ) -> HttpClient.Builder: ...
        def sslContext(
            self, sSLContext: javax.net.ssl.SSLContext
        ) -> HttpClient.Builder: ...
        def sslParameters(
            self, sSLParameters: javax.net.ssl.SSLParameters
        ) -> HttpClient.Builder: ...
        def version(self, version: HttpClient.Version) -> HttpClient.Builder: ...

    class Redirect(java.lang.Enum["HttpClient.Redirect"]):
        NEVER: typing.ClassVar[HttpClient.Redirect] = ...
        ALWAYS: typing.ClassVar[HttpClient.Redirect] = ...
        NORMAL: typing.ClassVar[HttpClient.Redirect] = ...
        _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_0__T], string: java.lang.String | str
        ) -> _valueOf_0__T: ...
        @typing.overload
        @staticmethod
        def valueOf(string: java.lang.String | str) -> HttpClient.Redirect: ...
        @staticmethod
        def values() -> typing.MutableSequence[HttpClient.Redirect]: ...

    class Version(java.lang.Enum["HttpClient.Version"]):
        HTTP_1_1: typing.ClassVar[HttpClient.Version] = ...
        HTTP_2: typing.ClassVar[HttpClient.Version] = ...
        _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_0__T], string: java.lang.String | str
        ) -> _valueOf_0__T: ...
        @typing.overload
        @staticmethod
        def valueOf(string: java.lang.String | str) -> HttpClient.Version: ...
        @staticmethod
        def values() -> typing.MutableSequence[HttpClient.Version]: ...

class HttpHeaders:
    def allValues(
        self, string: java.lang.String | str
    ) -> java.util.List[java.lang.String]: ...
    def equals(self, object: typing.Any) -> bool: ...
    def firstValue(
        self, string: java.lang.String | str
    ) -> java.util.Optional[java.lang.String]: ...
    def firstValueAsLong(
        self, string: java.lang.String | str
    ) -> java.util.OptionalLong: ...
    def hashCode(self) -> int: ...
    def map(
        self,
    ) -> java.util.Map[java.lang.String, java.util.List[java.lang.String]]: ...
    @staticmethod
    def of(
        map: java.util.Map[
            java.lang.String | str, java.util.List[java.lang.String | str]
        ]
        | typing.Mapping[
            java.lang.String | str, java.util.List[java.lang.String | str]
        ],
        biPredicate: java.util.function.BiPredicate[
            java.lang.String | str, java.lang.String | str
        ]
        | typing.Callable[[java.lang.String | str, java.lang.String | str], bool],
    ) -> HttpHeaders: ...
    def toString(self) -> java.lang.String: ...

class HttpRequest:
    def bodyPublisher(self) -> java.util.Optional[HttpRequest.BodyPublisher]: ...
    def equals(self, object: typing.Any) -> bool: ...
    def expectContinue(self) -> bool: ...
    def hashCode(self) -> int: ...
    def headers(self) -> HttpHeaders: ...
    def method(self) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def newBuilder() -> HttpRequest.Builder: ...
    @typing.overload
    @staticmethod
    def newBuilder(uRI: java.net.URI) -> HttpRequest.Builder: ...
    @typing.overload
    @staticmethod
    def newBuilder(
        httpRequest: HttpRequest,
        biPredicate: java.util.function.BiPredicate[
            java.lang.String | str, java.lang.String | str
        ]
        | typing.Callable[[java.lang.String | str, java.lang.String | str], bool],
    ) -> HttpRequest.Builder: ...
    def timeout(self) -> java.util.Optional[java.time.Duration]: ...
    def uri(self) -> java.net.URI: ...
    def version(self) -> java.util.Optional[HttpClient.Version]: ...
    class BodyPublisher(java.util.concurrent.Flow.Publisher[java.nio.ByteBuffer]):
        def contentLength(self) -> int: ...

    class BodyPublishers:
        @staticmethod
        def concat(
            *bodyPublisher: HttpRequest.BodyPublisher,
        ) -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def fromPublisher(
            publisher: java.util.concurrent.Flow.Publisher[java.nio.ByteBuffer]
            | typing.Callable[[java.util.concurrent.Flow.Subscriber[typing.Any]], None],
        ) -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def fromPublisher(
            publisher: java.util.concurrent.Flow.Publisher[java.nio.ByteBuffer]
            | typing.Callable[[java.util.concurrent.Flow.Subscriber[typing.Any]], None],
            long: int,
        ) -> HttpRequest.BodyPublisher: ...
        @staticmethod
        def noBody() -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def ofByteArray(
            byteArray: list[int] | jpype.JArray | bytes,
        ) -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def ofByteArray(
            byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
        ) -> HttpRequest.BodyPublisher: ...
        @staticmethod
        def ofByteArrays(
            iterable: java.lang.Iterable[list[int] | jpype.JArray | bytes]
            | typing.Sequence[list[int] | jpype.JArray | bytes]
            | set[list[int] | jpype.JArray | bytes]
            | typing.Callable[[], java.util.Iterator[typing.Any]],
        ) -> HttpRequest.BodyPublisher: ...
        @staticmethod
        def ofFile(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
        ) -> HttpRequest.BodyPublisher: ...
        @staticmethod
        def ofInputStream(
            supplier: java.util.function.Supplier[java.io.InputStream]
            | typing.Callable[[], java.io.InputStream],
        ) -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def ofString(string: java.lang.String | str) -> HttpRequest.BodyPublisher: ...
        @typing.overload
        @staticmethod
        def ofString(
            string: java.lang.String | str, charset: java.nio.charset.Charset
        ) -> HttpRequest.BodyPublisher: ...

    class Builder:
        def DELETE(self) -> HttpRequest.Builder: ...
        def GET(self) -> HttpRequest.Builder: ...
        def HEAD(self) -> HttpRequest.Builder: ...
        def POST(
            self, bodyPublisher: HttpRequest.BodyPublisher
        ) -> HttpRequest.Builder: ...
        def PUT(
            self, bodyPublisher: HttpRequest.BodyPublisher
        ) -> HttpRequest.Builder: ...
        def build(self) -> HttpRequest: ...
        def copy(self) -> HttpRequest.Builder: ...
        def expectContinue(self, boolean: bool) -> HttpRequest.Builder: ...
        def header(
            self, string: java.lang.String | str, string2: java.lang.String | str
        ) -> HttpRequest.Builder: ...
        def headers(self, *string: java.lang.String | str) -> HttpRequest.Builder: ...
        def method(
            self,
            string: java.lang.String | str,
            bodyPublisher: HttpRequest.BodyPublisher,
        ) -> HttpRequest.Builder: ...
        def setHeader(
            self, string: java.lang.String | str, string2: java.lang.String | str
        ) -> HttpRequest.Builder: ...
        def timeout(self, duration: java.time.Duration) -> HttpRequest.Builder: ...
        def uri(self, uRI: java.net.URI) -> HttpRequest.Builder: ...
        def version(self, version: HttpClient.Version) -> HttpRequest.Builder: ...

_HttpResponse__BodyHandler__T = typing.TypeVar("_HttpResponse__BodyHandler__T")  # <T>
_HttpResponse__BodySubscriber__T = typing.TypeVar(
    "_HttpResponse__BodySubscriber__T"
)  # <T>
_HttpResponse__PushPromiseHandler__T = typing.TypeVar(
    "_HttpResponse__PushPromiseHandler__T"
)  # <T>
_HttpResponse__T = typing.TypeVar("_HttpResponse__T")  # <T>

class HttpResponse(typing.Generic[_HttpResponse__T]):
    def body(self) -> _HttpResponse__T: ...
    def connectionLabel(self) -> java.util.Optional[java.lang.String]: ...
    def headers(self) -> HttpHeaders: ...
    def previousResponse(
        self,
    ) -> java.util.Optional[HttpResponse[_HttpResponse__T]]: ...
    def request(self) -> HttpRequest: ...
    def sslSession(self) -> java.util.Optional[javax.net.ssl.SSLSession]: ...
    def statusCode(self) -> int: ...
    def uri(self) -> java.net.URI: ...
    def version(self) -> HttpClient.Version: ...
    class BodyHandler(typing.Generic[_HttpResponse__BodyHandler__T]):
        def apply(
            self, responseInfo: HttpResponse.ResponseInfo
        ) -> HttpResponse.BodySubscriber[_HttpResponse__BodyHandler__T]: ...

    class BodyHandlers:
        _buffering__T = typing.TypeVar("_buffering__T")  # <T>
        @staticmethod
        def buffering(
            bodyHandler: HttpResponse.BodyHandler[_buffering__T]
            | typing.Callable[
                [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
            ],
            int: int,
        ) -> HttpResponse.BodyHandler[_buffering__T]: ...
        @staticmethod
        def discarding() -> HttpResponse.BodyHandler[None]: ...
        _fromLineSubscriber_1__S = typing.TypeVar(
            "_fromLineSubscriber_1__S", bound=java.util.concurrent.Flow.Subscriber
        )  # <S>
        _fromLineSubscriber_1__T = typing.TypeVar("_fromLineSubscriber_1__T")  # <T>
        @typing.overload
        @staticmethod
        def fromLineSubscriber(
            subscriber: java.util.concurrent.Flow.Subscriber[java.lang.String],
        ) -> HttpResponse.BodyHandler[None]: ...
        @typing.overload
        @staticmethod
        def fromLineSubscriber(
            s: _fromLineSubscriber_1__S,
            function: java.util.function.Function[
                _fromLineSubscriber_1__S, _fromLineSubscriber_1__T
            ]
            | typing.Callable[[_fromLineSubscriber_1__S], _fromLineSubscriber_1__T],
            string: java.lang.String | str,
        ) -> HttpResponse.BodyHandler[_fromLineSubscriber_1__T]: ...
        _fromSubscriber_1__S = typing.TypeVar(
            "_fromSubscriber_1__S", bound=java.util.concurrent.Flow.Subscriber
        )  # <S>
        _fromSubscriber_1__T = typing.TypeVar("_fromSubscriber_1__T")  # <T>
        @typing.overload
        @staticmethod
        def fromSubscriber(
            subscriber: java.util.concurrent.Flow.Subscriber[
                java.util.List[java.nio.ByteBuffer]
            ],
        ) -> HttpResponse.BodyHandler[None]: ...
        @typing.overload
        @staticmethod
        def fromSubscriber(
            s: _fromSubscriber_1__S,
            function: java.util.function.Function[
                _fromSubscriber_1__S, _fromSubscriber_1__T
            ]
            | typing.Callable[[_fromSubscriber_1__S], _fromSubscriber_1__T],
        ) -> HttpResponse.BodyHandler[_fromSubscriber_1__T]: ...
        _limiting__T = typing.TypeVar("_limiting__T")  # <T>
        @staticmethod
        def limiting(
            bodyHandler: HttpResponse.BodyHandler[_limiting__T]
            | typing.Callable[
                [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
            ],
            long: int,
        ) -> HttpResponse.BodyHandler[_limiting__T]: ...
        @staticmethod
        def ofByteArray() -> HttpResponse.BodyHandler[typing.MutableSequence[int]]: ...
        @staticmethod
        def ofByteArrayConsumer(
            consumer: java.util.function.Consumer[
                java.util.Optional[list[int] | jpype.JArray | bytes]
            ]
            | typing.Callable[
                [java.util.Optional[list[int] | jpype.JArray | bytes]], None
            ],
        ) -> HttpResponse.BodyHandler[None]: ...
        @typing.overload
        @staticmethod
        def ofFile(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
        ) -> HttpResponse.BodyHandler[java.nio.file.Path]: ...
        @typing.overload
        @staticmethod
        def ofFile(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
            *openOption: java.nio.file.OpenOption,
        ) -> HttpResponse.BodyHandler[java.nio.file.Path]: ...
        @staticmethod
        def ofFileDownload(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
            *openOption: java.nio.file.OpenOption,
        ) -> HttpResponse.BodyHandler[java.nio.file.Path]: ...
        @staticmethod
        def ofInputStream() -> HttpResponse.BodyHandler[java.io.InputStream]: ...
        @staticmethod
        def ofLines() -> HttpResponse.BodyHandler[
            java.util.stream.Stream[java.lang.String]
        ]: ...
        @staticmethod
        def ofPublisher() -> HttpResponse.BodyHandler[
            java.util.concurrent.Flow.Publisher[java.util.List[java.nio.ByteBuffer]]
        ]: ...
        @typing.overload
        @staticmethod
        def ofString() -> HttpResponse.BodyHandler[java.lang.String]: ...
        @typing.overload
        @staticmethod
        def ofString(
            charset: java.nio.charset.Charset,
        ) -> HttpResponse.BodyHandler[java.lang.String]: ...
        _replacing__U = typing.TypeVar("_replacing__U")  # <U>
        @staticmethod
        def replacing(u: _replacing__U) -> HttpResponse.BodyHandler[_replacing__U]: ...

    class BodySubscriber(
        java.util.concurrent.Flow.Subscriber[java.util.List[java.nio.ByteBuffer]],
        typing.Generic[_HttpResponse__BodySubscriber__T],
    ):
        def getBody(
            self,
        ) -> java.util.concurrent.CompletionStage[_HttpResponse__BodySubscriber__T]: ...

    class BodySubscribers:
        _buffering__T = typing.TypeVar("_buffering__T")  # <T>
        @staticmethod
        def buffering(
            bodySubscriber: HttpResponse.BodySubscriber[_buffering__T], int: int
        ) -> HttpResponse.BodySubscriber[_buffering__T]: ...
        @staticmethod
        def discarding() -> HttpResponse.BodySubscriber[None]: ...
        _fromLineSubscriber_1__S = typing.TypeVar(
            "_fromLineSubscriber_1__S", bound=java.util.concurrent.Flow.Subscriber
        )  # <S>
        _fromLineSubscriber_1__T = typing.TypeVar("_fromLineSubscriber_1__T")  # <T>
        @typing.overload
        @staticmethod
        def fromLineSubscriber(
            subscriber: java.util.concurrent.Flow.Subscriber[java.lang.String],
        ) -> HttpResponse.BodySubscriber[None]: ...
        @typing.overload
        @staticmethod
        def fromLineSubscriber(
            s: _fromLineSubscriber_1__S,
            function: java.util.function.Function[
                _fromLineSubscriber_1__S, _fromLineSubscriber_1__T
            ]
            | typing.Callable[[_fromLineSubscriber_1__S], _fromLineSubscriber_1__T],
            charset: java.nio.charset.Charset,
            string: java.lang.String | str,
        ) -> HttpResponse.BodySubscriber[_fromLineSubscriber_1__T]: ...
        _fromSubscriber_1__S = typing.TypeVar(
            "_fromSubscriber_1__S", bound=java.util.concurrent.Flow.Subscriber
        )  # <S>
        _fromSubscriber_1__T = typing.TypeVar("_fromSubscriber_1__T")  # <T>
        @typing.overload
        @staticmethod
        def fromSubscriber(
            subscriber: java.util.concurrent.Flow.Subscriber[
                java.util.List[java.nio.ByteBuffer]
            ],
        ) -> HttpResponse.BodySubscriber[None]: ...
        @typing.overload
        @staticmethod
        def fromSubscriber(
            s: _fromSubscriber_1__S,
            function: java.util.function.Function[
                _fromSubscriber_1__S, _fromSubscriber_1__T
            ]
            | typing.Callable[[_fromSubscriber_1__S], _fromSubscriber_1__T],
        ) -> HttpResponse.BodySubscriber[_fromSubscriber_1__T]: ...
        _limiting__T = typing.TypeVar("_limiting__T")  # <T>
        @staticmethod
        def limiting(
            bodySubscriber: HttpResponse.BodySubscriber[_limiting__T], long: int
        ) -> HttpResponse.BodySubscriber[_limiting__T]: ...
        _mapping__T = typing.TypeVar("_mapping__T")  # <T>
        _mapping__U = typing.TypeVar("_mapping__U")  # <U>
        @staticmethod
        def mapping(
            bodySubscriber: HttpResponse.BodySubscriber[_mapping__T],
            function: java.util.function.Function[_mapping__T, _mapping__U]
            | typing.Callable[[_mapping__T], _mapping__U],
        ) -> HttpResponse.BodySubscriber[_mapping__U]: ...
        @staticmethod
        def ofByteArray() -> HttpResponse.BodySubscriber[
            typing.MutableSequence[int]
        ]: ...
        @staticmethod
        def ofByteArrayConsumer(
            consumer: java.util.function.Consumer[
                java.util.Optional[list[int] | jpype.JArray | bytes]
            ]
            | typing.Callable[
                [java.util.Optional[list[int] | jpype.JArray | bytes]], None
            ],
        ) -> HttpResponse.BodySubscriber[None]: ...
        @typing.overload
        @staticmethod
        def ofFile(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
        ) -> HttpResponse.BodySubscriber[java.nio.file.Path]: ...
        @typing.overload
        @staticmethod
        def ofFile(
            path: java.nio.file.Path | jpype.protocol.SupportsPath,
            *openOption: java.nio.file.OpenOption,
        ) -> HttpResponse.BodySubscriber[java.nio.file.Path]: ...
        @staticmethod
        def ofInputStream() -> HttpResponse.BodySubscriber[java.io.InputStream]: ...
        @staticmethod
        def ofLines(
            charset: java.nio.charset.Charset,
        ) -> HttpResponse.BodySubscriber[java.util.stream.Stream[java.lang.String]]: ...
        @staticmethod
        def ofPublisher() -> HttpResponse.BodySubscriber[
            java.util.concurrent.Flow.Publisher[java.util.List[java.nio.ByteBuffer]]
        ]: ...
        @staticmethod
        def ofString(
            charset: java.nio.charset.Charset,
        ) -> HttpResponse.BodySubscriber[java.lang.String]: ...
        _replacing__U = typing.TypeVar("_replacing__U")  # <U>
        @staticmethod
        def replacing(
            u: _replacing__U,
        ) -> HttpResponse.BodySubscriber[_replacing__U]: ...

    class PushPromiseHandler(typing.Generic[_HttpResponse__PushPromiseHandler__T]):
        def applyPushPromise(
            self,
            httpRequest: HttpRequest,
            httpRequest2: HttpRequest,
            function: java.util.function.Function[
                HttpResponse.BodyHandler[_HttpResponse__PushPromiseHandler__T]
                | typing.Callable[
                    [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
                ],
                java.util.concurrent.CompletableFuture[
                    HttpResponse[_HttpResponse__PushPromiseHandler__T]
                ],
            ]
            | typing.Callable[
                [
                    HttpResponse.BodyHandler[_HttpResponse__PushPromiseHandler__T]
                    | typing.Callable[
                        [HttpResponse.ResponseInfo],
                        HttpResponse.BodySubscriber[typing.Any],
                    ]
                ],
                java.util.concurrent.CompletableFuture[
                    HttpResponse[_HttpResponse__PushPromiseHandler__T]
                ],
            ],
        ) -> None: ...
        _of__T = typing.TypeVar("_of__T")  # <T>
        @staticmethod
        def of(
            function: java.util.function.Function[
                HttpRequest,
                HttpResponse.BodyHandler[_of__T]
                | typing.Callable[
                    [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
                ],
            ]
            | typing.Callable[
                [HttpRequest],
                HttpResponse.BodyHandler[_of__T]
                | typing.Callable[
                    [HttpResponse.ResponseInfo], HttpResponse.BodySubscriber[typing.Any]
                ],
            ],
            concurrentMap: java.util.concurrent.ConcurrentMap[
                HttpRequest,
                java.util.concurrent.CompletableFuture[HttpResponse[_of__T]],
            ],
        ) -> HttpResponse.PushPromiseHandler[_of__T]: ...

    class ResponseInfo:
        def headers(self) -> HttpHeaders: ...
        def statusCode(self) -> int: ...
        def version(self) -> HttpClient.Version: ...

class HttpTimeoutException(java.io.IOException):
    def __init__(self, string: java.lang.String | str): ...

class WebSocket:
    NORMAL_CLOSURE: typing.ClassVar[int] = ...
    def abort(self) -> None: ...
    def getSubprotocol(self) -> java.lang.String: ...
    def isInputClosed(self) -> bool: ...
    def isOutputClosed(self) -> bool: ...
    def request(self, long: int) -> None: ...
    def sendBinary(
        self, byteBuffer: java.nio.ByteBuffer, boolean: bool
    ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
    def sendClose(
        self, int: int, string: java.lang.String | str
    ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
    def sendPing(
        self, byteBuffer: java.nio.ByteBuffer
    ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
    def sendPong(
        self, byteBuffer: java.nio.ByteBuffer
    ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
    def sendText(
        self, charSequence: java.lang.CharSequence | str, boolean: bool
    ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
    class Builder:
        def buildAsync(
            self, uRI: java.net.URI, listener: WebSocket.Listener
        ) -> java.util.concurrent.CompletableFuture[WebSocket]: ...
        def connectTimeout(self, duration: java.time.Duration) -> WebSocket.Builder: ...
        def header(
            self, string: java.lang.String | str, string2: java.lang.String | str
        ) -> WebSocket.Builder: ...
        def subprotocols(
            self, string: java.lang.String | str, *string2: java.lang.String | str
        ) -> WebSocket.Builder: ...

    class Listener:
        def onBinary(
            self, webSocket: WebSocket, byteBuffer: java.nio.ByteBuffer, boolean: bool
        ) -> java.util.concurrent.CompletionStage[typing.Any]: ...
        def onClose(
            self, webSocket: WebSocket, int: int, string: java.lang.String | str
        ) -> java.util.concurrent.CompletionStage[typing.Any]: ...
        def onError(
            self, webSocket: WebSocket, throwable: java.lang.Throwable
        ) -> None: ...
        def onOpen(self, webSocket: WebSocket) -> None: ...
        def onPing(
            self, webSocket: WebSocket, byteBuffer: java.nio.ByteBuffer
        ) -> java.util.concurrent.CompletionStage[typing.Any]: ...
        def onPong(
            self, webSocket: WebSocket, byteBuffer: java.nio.ByteBuffer
        ) -> java.util.concurrent.CompletionStage[typing.Any]: ...
        def onText(
            self,
            webSocket: WebSocket,
            charSequence: java.lang.CharSequence | str,
            boolean: bool,
        ) -> java.util.concurrent.CompletionStage[typing.Any]: ...

class WebSocketHandshakeException(java.io.IOException):
    def __init__(self, httpResponse: HttpResponse[typing.Any]): ...
    def getResponse(self) -> HttpResponse[typing.Any]: ...
    def initCause(
        self, throwable: java.lang.Throwable
    ) -> WebSocketHandshakeException: ...

class HttpConnectTimeoutException(HttpTimeoutException):
    def __init__(self, string: java.lang.String | str): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.net.http")``.

    HttpClient: type[HttpClient]
    HttpConnectTimeoutException: type[HttpConnectTimeoutException]
    HttpHeaders: type[HttpHeaders]
    HttpRequest: type[HttpRequest]
    HttpResponse: type[HttpResponse]
    HttpTimeoutException: type[HttpTimeoutException]
    WebSocket: type[WebSocket]
    WebSocketHandshakeException: type[WebSocketHandshakeException]
