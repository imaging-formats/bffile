import typing
from typing import Protocol

import java.lang
import java.security
import java.util
import java.util.jar
import jpype

class ClassDefinition:
    def __init__(
        self, class_: type[typing.Any], byteArray: list[int] | jpype.JArray | bytes
    ): ...
    def getDefinitionClass(self) -> type[typing.Any]: ...
    def getDefinitionClassFile(self) -> typing.MutableSequence[int]: ...

class ClassFileTransformer:
    @typing.overload
    def transform(
        self,
        classLoader: java.lang.ClassLoader,
        string: java.lang.String | str,
        class2: type[typing.Any],
        protectionDomain: java.security.ProtectionDomain,
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> typing.MutableSequence[int]: ...
    @typing.overload
    def transform(
        self,
        module: java.lang.Module,
        classLoader: java.lang.ClassLoader,
        string: java.lang.String | str,
        class2: type[typing.Any],
        protectionDomain: java.security.ProtectionDomain,
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> typing.MutableSequence[int]: ...

class IllegalClassFormatException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class Instrumentation:
    @typing.overload
    def addTransformer(self, classFileTransformer: ClassFileTransformer) -> None: ...
    @typing.overload
    def addTransformer(
        self, classFileTransformer: ClassFileTransformer, boolean: bool
    ) -> None: ...
    def appendToBootstrapClassLoaderSearch(
        self, jarFile: java.util.jar.JarFile
    ) -> None: ...
    def appendToSystemClassLoaderSearch(
        self, jarFile: java.util.jar.JarFile
    ) -> None: ...
    def getAllLoadedClasses(self) -> typing.MutableSequence[type]: ...
    def getInitiatedClasses(
        self, classLoader: java.lang.ClassLoader
    ) -> typing.MutableSequence[type]: ...
    def getObjectSize(self, object: typing.Any) -> int: ...
    def isModifiableClass(self, class_: type[typing.Any]) -> bool: ...
    def isModifiableModule(self, module: java.lang.Module) -> bool: ...
    def isNativeMethodPrefixSupported(self) -> bool: ...
    def isRedefineClassesSupported(self) -> bool: ...
    def isRetransformClassesSupported(self) -> bool: ...
    def redefineClasses(self, *classDefinition: ClassDefinition) -> None: ...
    def redefineModule(
        self,
        module: java.lang.Module,
        set: java.util.Set[java.lang.Module],
        map: java.util.Map[java.lang.String | str, java.util.Set[java.lang.Module]]
        | typing.Mapping[java.lang.String | str, java.util.Set[java.lang.Module]],
        map2: java.util.Map[java.lang.String | str, java.util.Set[java.lang.Module]]
        | typing.Mapping[java.lang.String | str, java.util.Set[java.lang.Module]],
        set2: java.util.Set[type[typing.Any]],
        map3: java.util.Map[type[typing.Any], java.util.List[type[typing.Any]]]
        | typing.Mapping[type[typing.Any], java.util.List[type[typing.Any]]],
    ) -> None: ...
    def removeTransformer(self, classFileTransformer: ClassFileTransformer) -> bool: ...
    def retransformClasses(self, *class_: type[typing.Any]) -> None: ...
    def setNativeMethodPrefix(
        self, classFileTransformer: ClassFileTransformer, string: java.lang.String | str
    ) -> None: ...

class UnmodifiableClassException(java.lang.Exception):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class UnmodifiableModuleException(java.lang.RuntimeException):
    @typing.overload
    def __init__(self): ...
    @typing.overload
    def __init__(self, string: java.lang.String | str): ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.lang.instrument")``.

    ClassDefinition: type[ClassDefinition]
    ClassFileTransformer: type[ClassFileTransformer]
    IllegalClassFormatException: type[IllegalClassFormatException]
    Instrumentation: type[Instrumentation]
    UnmodifiableClassException: type[UnmodifiableClassException]
    UnmodifiableModuleException: type[UnmodifiableModuleException]
