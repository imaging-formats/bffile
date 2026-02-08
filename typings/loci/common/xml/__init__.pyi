import typing
from typing import Protocol

import java.io
import java.lang
import java.util
import javax.xml.parsers
import javax.xml.transform
import jpype
import jpype.protocol
import loci.common
import org.w3c.dom
import org.w3c.dom.ls
import org.xml.sax
import org.xml.sax.helpers

class BaseHandler(org.xml.sax.helpers.DefaultHandler):
    def __init__(self): ...
    def resolveEntity(
        self, string: java.lang.String | str, string2: java.lang.String | str
    ) -> org.xml.sax.InputSource: ...

class LSInputI(org.w3c.dom.ls.LSInput):
    def __init__(self): ...
    def getBaseURI(self) -> java.lang.String: ...
    def getByteStream(self) -> java.io.InputStream: ...
    def getCertifiedText(self) -> bool: ...
    def getCharacterStream(self) -> java.io.Reader: ...
    def getEncoding(self) -> java.lang.String: ...
    def getPublicId(self) -> java.lang.String: ...
    def getStringData(self) -> java.lang.String: ...
    def getSystemId(self) -> java.lang.String: ...
    def setBaseURI(self, string: java.lang.String | str) -> None: ...
    def setByteStream(self, inputStream: java.io.InputStream) -> None: ...
    def setCertifiedText(self, boolean: bool) -> None: ...
    def setCharacterStream(self, reader: java.io.Reader) -> None: ...
    def setEncoding(self, string: java.lang.String | str) -> None: ...
    def setPublicId(self, string: java.lang.String | str) -> None: ...
    def setStringData(self, string: java.lang.String | str) -> None: ...
    def setSystemId(self, string: java.lang.String | str) -> None: ...

class ParserErrorHandler(org.xml.sax.ErrorHandler):
    def __init__(self): ...
    def error(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...
    def fatalError(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...
    def warning(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...

class ValidationErrorHandler(org.xml.sax.ErrorHandler):
    def __init__(self): ...
    def error(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...
    def fatalError(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...
    def getErrorCount(self) -> int: ...
    def ok(self) -> bool: ...
    def warning(self, sAXParseException: org.xml.sax.SAXParseException) -> None: ...

class XMLTools:
    @staticmethod
    def avoidUndeclaredNamespaces(
        string: java.lang.String | str,
    ) -> java.lang.String: ...
    @staticmethod
    def createBuilder() -> javax.xml.parsers.DocumentBuilder: ...
    @staticmethod
    def createDocument() -> org.w3c.dom.Document: ...
    @typing.overload
    @staticmethod
    def dumpXML(
        string: java.lang.String | str,
        document: org.w3c.dom.Document,
        element: org.w3c.dom.Element,
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def dumpXML(
        string: java.lang.String | str,
        document: org.w3c.dom.Document,
        element: org.w3c.dom.Element,
        boolean: bool,
    ) -> java.lang.String: ...
    @staticmethod
    def escapeXML(string: java.lang.String | str) -> java.lang.String: ...
    @staticmethod
    def getStylesheet(
        string: java.lang.String | str, class_: type[typing.Any]
    ) -> javax.xml.transform.Templates: ...
    @staticmethod
    def getXML(document: org.w3c.dom.Document) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def indentXML(string: java.lang.String | str) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def indentXML(
        string: java.lang.String | str, boolean: bool
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def indentXML(string: java.lang.String | str, int: int) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def indentXML(
        string: java.lang.String | str, int: int, boolean: bool
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def parseDOM(
        file: java.io.File | jpype.protocol.SupportsPath,
    ) -> org.w3c.dom.Document: ...
    @typing.overload
    @staticmethod
    def parseDOM(inputStream: java.io.InputStream) -> org.w3c.dom.Document: ...
    @typing.overload
    @staticmethod
    def parseDOM(string: java.lang.String | str) -> org.w3c.dom.Document: ...
    @typing.overload
    @staticmethod
    def parseXML(
        string: java.lang.String | str,
    ) -> java.util.Hashtable[java.lang.String, java.lang.String]: ...
    @typing.overload
    @staticmethod
    def parseXML(
        byteArray: list[int] | jpype.JArray | bytes,
        defaultHandler: org.xml.sax.helpers.DefaultHandler,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def parseXML(
        inputStream: java.io.InputStream,
        defaultHandler: org.xml.sax.helpers.DefaultHandler,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def parseXML(
        string: java.lang.String | str,
        defaultHandler: org.xml.sax.helpers.DefaultHandler,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def parseXML(
        randomAccessInputStream: loci.common.RandomAccessInputStream,
        defaultHandler: org.xml.sax.helpers.DefaultHandler,
    ) -> None: ...
    @staticmethod
    def sanitizeXML(string: java.lang.String | str) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def transformXML(
        string: java.lang.String | str, templates: javax.xml.transform.Templates
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def transformXML(
        source: javax.xml.transform.Source, templates: javax.xml.transform.Templates
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def validateXML(string: java.lang.String | str) -> bool: ...
    @typing.overload
    @staticmethod
    def validateXML(
        string: java.lang.String | str, string2: java.lang.String | str
    ) -> bool: ...
    @typing.overload
    @staticmethod
    def validateXML(
        string: java.lang.String | str,
        string2: java.lang.String | str,
        schemaReader: XMLTools.SchemaReader | typing.Callable,
    ) -> bool: ...
    @typing.overload
    @staticmethod
    def writeXML(
        outputStream: java.io.OutputStream, document: org.w3c.dom.Document
    ) -> None: ...
    @typing.overload
    @staticmethod
    def writeXML(
        outputStream: java.io.OutputStream,
        document: org.w3c.dom.Document,
        boolean: bool,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def writeXML(
        result: javax.xml.transform.Result,
        document: org.w3c.dom.Document,
        boolean: bool,
    ) -> None: ...
    class SchemaReader:
        def getSchemaAsStream(
            self, string: java.lang.String | str
        ) -> java.io.InputStream: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.common.xml")``.

    BaseHandler: type[BaseHandler]
    LSInputI: type[LSInputI]
    ParserErrorHandler: type[ParserErrorHandler]
    ValidationErrorHandler: type[ValidationErrorHandler]
    XMLTools: type[XMLTools]
