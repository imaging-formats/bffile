import typing
from typing import Protocol

import java.io
import java.lang
import java.util
import jpype

class ArrayUtil:
    def __init__(self): ...
    @staticmethod
    def arrayMoveWithin(
        objectArray: list[typing.Any] | jpype.JArray, int: int, int2: int, int3: int
    ) -> None: ...
    @staticmethod
    def arraycopy(
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        byteArray2: list[int] | jpype.JArray | bytes,
        int2: int,
        int3: int,
    ) -> None: ...

class BinaryTree(java.util.AbstractMap):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, map: java.util.Map | typing.Mapping): ...
    def clear(self) -> None: ...
    def containsKey(self, object: typing.Any) -> bool: ...
    def containsValue(self, object: typing.Any) -> bool: ...
    def entrySet(self) -> java.util.Set: ...
    def entrySetByValue(self) -> java.util.Set: ...
    def get(self, object: typing.Any) -> typing.Any: ...
    def getKeyForValue(self, object: typing.Any) -> typing.Any: ...
    def keySet(self) -> java.util.Set: ...
    def keySetByValue(self) -> java.util.Set: ...
    def put(self, object: typing.Any, object2: typing.Any) -> typing.Any: ...
    @typing.overload
    def remove(self, object: typing.Any, object2: typing.Any) -> bool: ...
    @typing.overload
    def remove(self, object: typing.Any) -> typing.Any: ...
    def removeValue(self, object: typing.Any) -> typing.Any: ...
    def size(self) -> int: ...
    def values(self) -> java.util.Collection: ...
    def valuesByValue(self) -> java.util.Collection: ...

class BitField:
    def __init__(self, int: int): ...
    def clear(self, int: int) -> int: ...
    def clearByte(self, byte: int) -> int: ...
    def clearShort(self, short: int) -> int: ...
    def getRawValue(self, int: int) -> int: ...
    def getShortRawValue(self, short: int) -> int: ...
    def getShortValue(self, short: int) -> int: ...
    def getValue(self, int: int) -> int: ...
    def isAllSet(self, int: int) -> bool: ...
    def isSet(self, int: int) -> bool: ...
    def set(self, int: int) -> int: ...
    def setBoolean(self, int: int, boolean: bool) -> int: ...
    def setByte(self, byte: int) -> int: ...
    def setByteBoolean(self, byte: int, boolean: bool) -> int: ...
    def setShort(self, short: int) -> int: ...
    def setShortBoolean(self, short: int, boolean: bool) -> int: ...
    def setShortValue(self, short: int, short2: int) -> int: ...
    def setValue(self, int: int, int2: int) -> int: ...

class BitFieldFactory:
    def __init__(self): ...
    @staticmethod
    def getInstance(int: int) -> BitField: ...

class BlockingInputStream(java.io.InputStream):
    def __init__(self, inputStream: java.io.InputStream): ...
    def available(self) -> int: ...
    def close(self) -> None: ...
    def mark(self, int: int) -> None: ...
    def markSupported(self) -> bool: ...
    @typing.overload
    def read(self) -> int: ...
    @typing.overload
    def read(self, byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    def read(
        self, byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> int: ...
    def reset(self) -> None: ...
    def skip(self, long: int) -> int: ...

class DoubleList:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, doubleList: DoubleList): ...
    @typing.overload
    def add(self, double: float) -> bool: ...
    @typing.overload
    def add(self, int: int, double: float) -> None: ...
    @typing.overload
    def addAll(self, int: int, doubleList: DoubleList) -> bool: ...
    @typing.overload
    def addAll(self, doubleList: DoubleList) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, double: float) -> bool: ...
    def containsAll(self, doubleList: DoubleList) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def get(self, int: int) -> float: ...
    def hashCode(self) -> int: ...
    def indexOf(self, double: float) -> int: ...
    def isEmpty(self) -> bool: ...
    def lastIndexOf(self, double: float) -> int: ...
    def remove(self, int: int) -> float: ...
    def removeAll(self, doubleList: DoubleList) -> bool: ...
    def removeValue(self, double: float) -> bool: ...
    def retainAll(self, doubleList: DoubleList) -> bool: ...
    def set(self, int: int, double: float) -> float: ...
    def size(self) -> int: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[float]: ...
    @typing.overload
    def toArray(
        self, doubleArray: list[float] | jpype.JArray
    ) -> typing.MutableSequence[float]: ...

class DoubleList2d:
    def __init__(self): ...
    def get(self, int: int, int2: int) -> float: ...
    def set(self, int: int, int2: int, double: float) -> None: ...

class DrawingDump:
    def __init__(self): ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...

class FixedField:
    def readFromBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def readFromStream(self, inputStream: java.io.InputStream) -> None: ...
    def toString(self) -> java.lang.String: ...
    def writeToBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...

class HexDump:
    EOL: typing.ClassVar[java.lang.String] = ...
    @typing.overload
    @staticmethod
    def dump(
        byteArray: list[int] | jpype.JArray | bytes, long: int, int: int
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def dump(
        byteArray: list[int] | jpype.JArray | bytes,
        long: int,
        outputStream: java.io.OutputStream,
        int: int,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def dump(
        byteArray: list[int] | jpype.JArray | bytes,
        long: int,
        outputStream: java.io.OutputStream,
        int: int,
        int2: int,
    ) -> None: ...
    @typing.overload
    @staticmethod
    def dump(
        inputStream: java.io.InputStream,
        printStream: java.io.PrintStream,
        int: int,
        int2: int,
    ) -> None: ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...
    @typing.overload
    @staticmethod
    def toHex(byte: int) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(byteArray: list[int] | jpype.JArray | bytes) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(
        byteArray: list[int] | jpype.JArray | bytes, int: int
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(int: int) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(long: int) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(short: int) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def toHex(shortArray: list[int] | jpype.JArray) -> java.lang.String: ...

class HexRead:
    def __init__(self): ...
    @typing.overload
    @staticmethod
    def readData(
        inputStream: java.io.InputStream, int: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def readData(string: java.lang.String | str) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def readData(
        string: java.lang.String | str, string2: java.lang.String | str
    ) -> typing.MutableSequence[int]: ...
    @staticmethod
    def readFromString(
        string: java.lang.String | str,
    ) -> typing.MutableSequence[int]: ...

class IOUtils:
    @typing.overload
    @staticmethod
    def readFully(
        inputStream: java.io.InputStream, byteArray: list[int] | jpype.JArray | bytes
    ) -> int: ...
    @typing.overload
    @staticmethod
    def readFully(
        inputStream: java.io.InputStream,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        int2: int,
    ) -> int: ...

class IntList:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    @typing.overload
    def __init__(self, intList: IntList): ...
    @typing.overload
    def add(self, int: int) -> bool: ...
    @typing.overload
    def add(self, int: int, int2: int) -> None: ...
    @typing.overload
    def addAll(self, int: int, intList: IntList) -> bool: ...
    @typing.overload
    def addAll(self, intList: IntList) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, int: int) -> bool: ...
    def containsAll(self, intList: IntList) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def get(self, int: int) -> int: ...
    def hashCode(self) -> int: ...
    def indexOf(self, int: int) -> int: ...
    def isEmpty(self) -> bool: ...
    def lastIndexOf(self, int: int) -> int: ...
    def remove(self, int: int) -> int: ...
    def removeAll(self, intList: IntList) -> bool: ...
    def removeValue(self, int: int) -> bool: ...
    def retainAll(self, intList: IntList) -> bool: ...
    def set(self, int: int, int2: int) -> int: ...
    def size(self) -> int: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def toArray(
        self, intArray: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...

class IntList2d:
    def __init__(self): ...
    def get(self, int: int, int2: int) -> int: ...
    def isAllocated(self, int: int, int2: int) -> bool: ...
    def set(self, int: int, int2: int, int3: int) -> None: ...

class IntMapper:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    def add(self, object: typing.Any) -> bool: ...
    def get(self, int: int) -> typing.Any: ...
    def getIndex(self, object: typing.Any) -> int: ...
    def iterator(self) -> java.util.Iterator: ...
    def size(self) -> int: ...

class List2d:
    def __init__(self): ...
    def get(self, int: int, int2: int) -> typing.Any: ...
    def set(self, int: int, int2: int, object: typing.Any) -> None: ...

class LittleEndianConsts:
    BYTE_SIZE: typing.ClassVar[int] = ...
    SHORT_SIZE: typing.ClassVar[int] = ...
    INT_SIZE: typing.ClassVar[int] = ...
    LONG_SIZE: typing.ClassVar[int] = ...
    DOUBLE_SIZE: typing.ClassVar[int] = ...

class POILogFactory:
    @typing.overload
    @staticmethod
    def getLogger(class_: type) -> POILogger: ...
    @typing.overload
    @staticmethod
    def getLogger(string: java.lang.String | str) -> POILogger: ...

class POILogger:
    DEBUG: typing.ClassVar[int] = ...
    INFO: typing.ClassVar[int] = ...
    WARN: typing.ClassVar[int] = ...
    ERROR: typing.ClassVar[int] = ...
    FATAL: typing.ClassVar[int] = ...
    def check(self, int: int) -> bool: ...
    def initialize(self, string: java.lang.String | str) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any, object2: typing.Any) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, object2: typing.Any, object3: typing.Any
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, throwable: java.lang.Throwable
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self, int: int, string: java.lang.String | str, object: typing.Any
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...

class ShortList:
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, shortList: ShortList): ...
    @typing.overload
    def add(self, short: int) -> bool: ...
    @typing.overload
    def add(self, int: int, short: int) -> None: ...
    @typing.overload
    def addAll(self, int: int, shortList: ShortList) -> bool: ...
    @typing.overload
    def addAll(self, shortList: ShortList) -> bool: ...
    def clear(self) -> None: ...
    def contains(self, short: int) -> bool: ...
    def containsAll(self, shortList: ShortList) -> bool: ...
    def equals(self, object: typing.Any) -> bool: ...
    def get(self, int: int) -> int: ...
    def hashCode(self) -> int: ...
    def indexOf(self, short: int) -> int: ...
    def isEmpty(self) -> bool: ...
    def lastIndexOf(self, short: int) -> int: ...
    def remove(self, int: int) -> int: ...
    def removeAll(self, shortList: ShortList) -> bool: ...
    def removeValue(self, short: int) -> bool: ...
    def retainAll(self, shortList: ShortList) -> bool: ...
    def set(self, int: int, short: int) -> int: ...
    def size(self) -> int: ...
    @typing.overload
    def toArray(self) -> typing.MutableSequence[int]: ...
    @typing.overload
    def toArray(
        self, shortArray: list[int] | jpype.JArray
    ) -> typing.MutableSequence[int]: ...

class StringUtil:
    @staticmethod
    def format(
        string: java.lang.String | str, objectArray: list[typing.Any] | jpype.JArray
    ) -> java.lang.String: ...
    @staticmethod
    def getFromCompressedUnicode(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def getFromUnicodeBE(
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def getFromUnicodeBE(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def getFromUnicodeLE(
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> java.lang.String: ...
    @typing.overload
    @staticmethod
    def getFromUnicodeLE(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> java.lang.String: ...
    @staticmethod
    def getPreferredEncoding() -> java.lang.String: ...
    @staticmethod
    def hasMultibyte(string: java.lang.String | str) -> bool: ...
    @staticmethod
    def isUnicodeString(string: java.lang.String | str) -> bool: ...
    @staticmethod
    def putCompressedUnicode(
        string: java.lang.String | str,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
    ) -> None: ...
    @staticmethod
    def putUnicodeBE(
        string: java.lang.String | str,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
    ) -> None: ...
    @staticmethod
    def putUnicodeLE(
        string: java.lang.String | str,
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
    ) -> None: ...

class TempFile:
    def __init__(self): ...
    @staticmethod
    def createTempFile(
        string: java.lang.String | str, string2: java.lang.String | str
    ) -> java.io.File: ...

class ByteField(FixedField):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, byte: int): ...
    @typing.overload
    def __init__(
        self, int: int, byte: int, byteArray: list[int] | jpype.JArray | bytes
    ): ...
    @typing.overload
    def __init__(self, int: int, byteArray: list[int] | jpype.JArray | bytes): ...
    def get(self) -> int: ...
    def readFromBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def readFromStream(self, inputStream: java.io.InputStream) -> None: ...
    @typing.overload
    def set(self, byte: int) -> None: ...
    @typing.overload
    def set(self, byte: int, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def toString(self) -> java.lang.String: ...
    def writeToBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...

class CommonsLogger(POILogger):
    def __init__(self): ...
    def check(self, int: int) -> bool: ...
    def initialize(self, string: java.lang.String | str) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any, object2: typing.Any) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, object2: typing.Any, object3: typing.Any
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, throwable: java.lang.Throwable
    ) -> None: ...

class IntegerField(FixedField):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, byteArray: list[int] | jpype.JArray | bytes): ...
    @typing.overload
    def __init__(self, int: int, int2: int): ...
    @typing.overload
    def __init__(
        self, int: int, int2: int, byteArray: list[int] | jpype.JArray | bytes
    ): ...
    def get(self) -> int: ...
    def readFromBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def readFromStream(self, inputStream: java.io.InputStream) -> None: ...
    @typing.overload
    def set(self, int: int) -> None: ...
    @typing.overload
    def set(self, int: int, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def toString(self) -> java.lang.String: ...
    def writeToBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...

class LittleEndian(LittleEndianConsts):
    @staticmethod
    def getByteArray(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def getDouble(byteArray: list[int] | jpype.JArray | bytes) -> float: ...
    @typing.overload
    @staticmethod
    def getDouble(byteArray: list[int] | jpype.JArray | bytes, int: int) -> float: ...
    @typing.overload
    @staticmethod
    def getInt(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getInt(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @typing.overload
    @staticmethod
    def getLong(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getLong(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @typing.overload
    @staticmethod
    def getShort(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getShort(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @staticmethod
    def getShortArray(
        byteArray: list[int] | jpype.JArray | bytes, int: int
    ) -> typing.MutableSequence[int]: ...
    @staticmethod
    def getSimpleShortArray(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    @staticmethod
    def getUInt(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getUInt(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @staticmethod
    def getULong(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @typing.overload
    @staticmethod
    def getUShort(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getUShort(byteArray: list[int] | jpype.JArray | bytes, int: int) -> int: ...
    @typing.overload
    @staticmethod
    def getUnsignedByte(byteArray: list[int] | jpype.JArray | bytes) -> int: ...
    @typing.overload
    @staticmethod
    def getUnsignedByte(
        byteArray: list[int] | jpype.JArray | bytes, int: int
    ) -> int: ...
    @typing.overload
    @staticmethod
    def putDouble(
        byteArray: list[int] | jpype.JArray | bytes, double: float
    ) -> None: ...
    @typing.overload
    @staticmethod
    def putDouble(
        byteArray: list[int] | jpype.JArray | bytes, int: int, double: float
    ) -> None: ...
    @typing.overload
    @staticmethod
    def putInt(byteArray: list[int] | jpype.JArray | bytes, int: int) -> None: ...
    @typing.overload
    @staticmethod
    def putInt(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> None: ...
    @typing.overload
    @staticmethod
    def putLong(
        byteArray: list[int] | jpype.JArray | bytes, int: int, long: int
    ) -> None: ...
    @typing.overload
    @staticmethod
    def putLong(byteArray: list[int] | jpype.JArray | bytes, long: int) -> None: ...
    @typing.overload
    @staticmethod
    def putShort(
        byteArray: list[int] | jpype.JArray | bytes, int: int, short: int
    ) -> None: ...
    @typing.overload
    @staticmethod
    def putShort(byteArray: list[int] | jpype.JArray | bytes, short: int) -> None: ...
    @staticmethod
    def putShortArray(
        byteArray: list[int] | jpype.JArray | bytes,
        int: int,
        shortArray: list[int] | jpype.JArray,
    ) -> None: ...
    @staticmethod
    def putUShort(
        byteArray: list[int] | jpype.JArray | bytes, int: int, int2: int
    ) -> None: ...
    @staticmethod
    def readFromStream(
        inputStream: java.io.InputStream, int: int
    ) -> typing.MutableSequence[int]: ...
    @staticmethod
    def readInt(inputStream: java.io.InputStream) -> int: ...
    @staticmethod
    def readLong(inputStream: java.io.InputStream) -> int: ...
    @staticmethod
    def readShort(inputStream: java.io.InputStream) -> int: ...
    @staticmethod
    def ubyteToInt(byte: int) -> int: ...
    class BufferUnderrunException(java.io.IOException): ...

class LongField(FixedField):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, byteArray: list[int] | jpype.JArray | bytes): ...
    @typing.overload
    def __init__(self, int: int, long: int): ...
    @typing.overload
    def __init__(
        self, int: int, long: int, byteArray: list[int] | jpype.JArray | bytes
    ): ...
    def get(self) -> int: ...
    def readFromBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def readFromStream(self, inputStream: java.io.InputStream) -> None: ...
    @typing.overload
    def set(self, long: int) -> None: ...
    @typing.overload
    def set(self, long: int, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def toString(self) -> java.lang.String: ...
    def writeToBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...

class NullLogger(POILogger):
    def __init__(self): ...
    def check(self, int: int) -> bool: ...
    def initialize(self, string: java.lang.String | str) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any, object2: typing.Any) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, object2: typing.Any, object3: typing.Any
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, throwable: java.lang.Throwable
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self, int: int, string: java.lang.String | str, object: typing.Any
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
    ) -> None: ...
    @typing.overload
    def logFormatted(
        self,
        int: int,
        string: java.lang.String | str,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...

class ShortField(FixedField):
    @typing.overload
    def __init__(self, int: int): ...
    @typing.overload
    def __init__(self, int: int, byteArray: list[int] | jpype.JArray | bytes): ...
    @typing.overload
    def __init__(self, int: int, short: int): ...
    @typing.overload
    def __init__(
        self, int: int, short: int, byteArray: list[int] | jpype.JArray | bytes
    ): ...
    def get(self) -> int: ...
    def readFromBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def readFromStream(self, inputStream: java.io.InputStream) -> None: ...
    @typing.overload
    def set(self, short: int) -> None: ...
    @typing.overload
    def set(self, short: int, byteArray: list[int] | jpype.JArray | bytes) -> None: ...
    def toString(self) -> java.lang.String: ...
    def writeToBytes(self, byteArray: list[int] | jpype.JArray | bytes) -> None: ...

class SystemOutLogger(POILogger):
    def __init__(self): ...
    def check(self, int: int) -> bool: ...
    def initialize(self, string: java.lang.String | str) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any, object2: typing.Any) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, object2: typing.Any, object3: typing.Any
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        object8: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        object7: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        object6: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        object5: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        object4: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        object3: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self,
        int: int,
        object: typing.Any,
        object2: typing.Any,
        throwable: java.lang.Throwable,
    ) -> None: ...
    @typing.overload
    def log(
        self, int: int, object: typing.Any, throwable: java.lang.Throwable
    ) -> None: ...
    @typing.overload
    def log(self, int: int, object: typing.Any) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.util")``.

    ArrayUtil: type[ArrayUtil]
    BinaryTree: type[BinaryTree]
    BitField: type[BitField]
    BitFieldFactory: type[BitFieldFactory]
    BlockingInputStream: type[BlockingInputStream]
    ByteField: type[ByteField]
    CommonsLogger: type[CommonsLogger]
    DoubleList: type[DoubleList]
    DoubleList2d: type[DoubleList2d]
    DrawingDump: type[DrawingDump]
    FixedField: type[FixedField]
    HexDump: type[HexDump]
    HexRead: type[HexRead]
    IOUtils: type[IOUtils]
    IntList: type[IntList]
    IntList2d: type[IntList2d]
    IntMapper: type[IntMapper]
    IntegerField: type[IntegerField]
    List2d: type[List2d]
    LittleEndian: type[LittleEndian]
    LittleEndianConsts: type[LittleEndianConsts]
    LongField: type[LongField]
    NullLogger: type[NullLogger]
    POILogFactory: type[POILogFactory]
    POILogger: type[POILogger]
    ShortField: type[ShortField]
    ShortList: type[ShortList]
    StringUtil: type[StringUtil]
    SystemOutLogger: type[SystemOutLogger]
    TempFile: type[TempFile]
