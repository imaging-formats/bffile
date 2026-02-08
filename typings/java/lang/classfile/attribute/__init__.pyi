import typing
from typing import Protocol

import java
import java.lang
import java.lang.classfile
import java.lang.classfile.constantpool
import java.lang.constant
import java.lang.reflect
import java.util
import java.util.function
import jpype

class AnnotationDefaultAttribute(
    java.lang.classfile.Attribute["AnnotationDefaultAttribute"],
    java.lang.classfile.MethodElement,
):
    def defaultValue(self) -> java.lang.classfile.AnnotationValue: ...
    @staticmethod
    def of(
        annotationValue: java.lang.classfile.AnnotationValue,
    ) -> AnnotationDefaultAttribute: ...

class BootstrapMethodsAttribute(
    java.lang.classfile.Attribute["BootstrapMethodsAttribute"]
):
    def bootstrapMethods(
        self,
    ) -> java.util.List[java.lang.classfile.BootstrapMethodEntry]: ...
    def bootstrapMethodsSize(self) -> int: ...

class CharacterRangeInfo:
    def characterRangeEnd(self) -> int: ...
    def characterRangeStart(self) -> int: ...
    def endPc(self) -> int: ...
    def flags(self) -> int: ...
    @staticmethod
    def of(
        int: int, int2: int, int3: int, int4: int, int5: int
    ) -> CharacterRangeInfo: ...
    def startPc(self) -> int: ...

class CharacterRangeTableAttribute(
    java.lang.classfile.Attribute["CharacterRangeTableAttribute"]
):
    def characterRangeTable(self) -> java.util.List[CharacterRangeInfo]: ...
    @staticmethod
    def of(
        list: java.util.List[CharacterRangeInfo],
    ) -> CharacterRangeTableAttribute: ...

class CodeAttribute(
    java.lang.classfile.Attribute["CodeAttribute"], java.lang.classfile.CodeModel
):
    def codeArray(self) -> typing.MutableSequence[int]: ...
    def codeLength(self) -> int: ...
    def labelToBci(self, label: java.lang.classfile.Label) -> int: ...
    def maxLocals(self) -> int: ...
    def maxStack(self) -> int: ...

class CompilationIDAttribute(
    java.lang.classfile.Attribute["CompilationIDAttribute"],
    java.lang.classfile.ClassElement,
):
    def compilationId(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    @typing.overload
    @staticmethod
    def of(string: java.lang.String | str) -> CompilationIDAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> CompilationIDAttribute: ...

class ConstantValueAttribute(
    java.lang.classfile.Attribute["ConstantValueAttribute"],
    java.lang.classfile.FieldElement,
):
    def constant(self) -> java.lang.classfile.constantpool.ConstantValueEntry: ...
    @typing.overload
    @staticmethod
    def of(
        constantValueEntry: java.lang.classfile.constantpool.ConstantValueEntry,
    ) -> ConstantValueAttribute: ...
    @typing.overload
    @staticmethod
    def of(constantDesc: java.lang.constant.ConstantDesc) -> ConstantValueAttribute: ...

class DeprecatedAttribute(
    java.lang.classfile.Attribute["DeprecatedAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
):
    @staticmethod
    def of() -> DeprecatedAttribute: ...

class EnclosingMethodAttribute(
    java.lang.classfile.Attribute["EnclosingMethodAttribute"],
    java.lang.classfile.ClassElement,
):
    def enclosingClass(self) -> java.lang.classfile.constantpool.ClassEntry: ...
    def enclosingMethod(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.NameAndTypeEntry]: ...
    def enclosingMethodName(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...
    def enclosingMethodType(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...
    def enclosingMethodTypeSymbol(
        self,
    ) -> java.util.Optional[java.lang.constant.MethodTypeDesc]: ...
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
        optional: java.util.Optional[java.lang.classfile.constantpool.NameAndTypeEntry],
    ) -> EnclosingMethodAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        classDesc: java.lang.constant.ClassDesc,
        optional: java.util.Optional[java.lang.String | str],
        optional2: java.util.Optional[java.lang.constant.MethodTypeDesc],
    ) -> EnclosingMethodAttribute: ...

class ExceptionsAttribute(
    java.lang.classfile.Attribute["ExceptionsAttribute"],
    java.lang.classfile.MethodElement,
):
    def exceptions(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ClassEntry]: ...
    @typing.overload
    @staticmethod
    def of(
        *classEntry: java.lang.classfile.constantpool.ClassEntry,
    ) -> ExceptionsAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.constantpool.ClassEntry],
    ) -> ExceptionsAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(*classDesc: java.lang.constant.ClassDesc) -> ExceptionsAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(
        list: java.util.List[java.lang.constant.ClassDesc],
    ) -> ExceptionsAttribute: ...

class InnerClassInfo:
    def flags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def flagsMask(self) -> int: ...
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    def innerClass(self) -> java.lang.classfile.constantpool.ClassEntry: ...
    def innerName(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
        optional: java.util.Optional[java.lang.classfile.constantpool.ClassEntry],
        optional2: java.util.Optional[java.lang.classfile.constantpool.Utf8Entry],
        int: int,
    ) -> InnerClassInfo: ...
    @typing.overload
    @staticmethod
    def of(
        classDesc: java.lang.constant.ClassDesc,
        optional: java.util.Optional[java.lang.constant.ClassDesc],
        optional2: java.util.Optional[java.lang.String | str],
        int: int,
    ) -> InnerClassInfo: ...
    @typing.overload
    @staticmethod
    def of(
        classDesc: java.lang.constant.ClassDesc,
        optional: java.util.Optional[java.lang.constant.ClassDesc],
        optional2: java.util.Optional[java.lang.String | str],
        *accessFlag: java.lang.reflect.AccessFlag,
    ) -> InnerClassInfo: ...
    def outerClass(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.ClassEntry]: ...

class InnerClassesAttribute(
    java.lang.classfile.Attribute["InnerClassesAttribute"],
    java.lang.classfile.ClassElement,
):
    def classes(self) -> java.util.List[InnerClassInfo]: ...
    @typing.overload
    @staticmethod
    def of(*innerClassInfo: InnerClassInfo) -> InnerClassesAttribute: ...
    @typing.overload
    @staticmethod
    def of(list: java.util.List[InnerClassInfo]) -> InnerClassesAttribute: ...

class LineNumberInfo:
    def lineNumber(self) -> int: ...
    @staticmethod
    def of(int: int, int2: int) -> LineNumberInfo: ...
    def startPc(self) -> int: ...

class LineNumberTableAttribute(
    java.lang.classfile.Attribute["LineNumberTableAttribute"]
):
    def lineNumbers(self) -> java.util.List[LineNumberInfo]: ...
    @staticmethod
    def of(list: java.util.List[LineNumberInfo]) -> LineNumberTableAttribute: ...

class LocalVariableInfo:
    def length(self) -> int: ...
    def name(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def slot(self) -> int: ...
    def startPc(self) -> int: ...
    def type(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def typeSymbol(self) -> java.lang.constant.ClassDesc: ...

class LocalVariableTableAttribute(
    java.lang.classfile.Attribute["LocalVariableTableAttribute"]
):
    def localVariables(self) -> java.util.List[LocalVariableInfo]: ...
    @staticmethod
    def of(list: java.util.List[LocalVariableInfo]) -> LocalVariableTableAttribute: ...

class LocalVariableTypeInfo:
    def length(self) -> int: ...
    def name(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def signature(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def slot(self) -> int: ...
    def startPc(self) -> int: ...

class LocalVariableTypeTableAttribute(
    java.lang.classfile.Attribute["LocalVariableTypeTableAttribute"]
):
    def localVariableTypes(self) -> java.util.List[LocalVariableTypeInfo]: ...
    @staticmethod
    def of(
        list: java.util.List[LocalVariableTypeInfo],
    ) -> LocalVariableTypeTableAttribute: ...

class MethodParameterInfo:
    def flags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def flagsMask(self) -> int: ...
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    def name(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...
    @typing.overload
    @staticmethod
    def of(
        optional: java.util.Optional[java.lang.classfile.constantpool.Utf8Entry],
        int: int,
    ) -> MethodParameterInfo: ...
    @typing.overload
    @staticmethod
    def of(
        optional: java.util.Optional[java.lang.String | str],
        *accessFlag: java.lang.reflect.AccessFlag,
    ) -> MethodParameterInfo: ...
    @staticmethod
    def ofParameter(
        optional: java.util.Optional[java.lang.String | str], int: int
    ) -> MethodParameterInfo: ...

class MethodParametersAttribute(
    java.lang.classfile.Attribute["MethodParametersAttribute"],
    java.lang.classfile.MethodElement,
):
    @typing.overload
    @staticmethod
    def of(*methodParameterInfo: MethodParameterInfo) -> MethodParametersAttribute: ...
    @typing.overload
    @staticmethod
    def of(list: java.util.List[MethodParameterInfo]) -> MethodParametersAttribute: ...
    def parameters(self) -> java.util.List[MethodParameterInfo]: ...

class ModuleAttribute(
    java.lang.classfile.Attribute["ModuleAttribute"], java.lang.classfile.ClassElement
):
    def exports(self) -> java.util.List[ModuleExportInfo]: ...
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    def moduleFlags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def moduleFlagsMask(self) -> int: ...
    def moduleName(self) -> java.lang.classfile.constantpool.ModuleEntry: ...
    def moduleVersion(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...
    @typing.overload
    @staticmethod
    def of(
        moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
        int: int,
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
        collection: java.util.Collection[ModuleRequireInfo]
        | typing.Sequence[ModuleRequireInfo]
        | set[ModuleRequireInfo],
        collection2: java.util.Collection[ModuleExportInfo]
        | typing.Sequence[ModuleExportInfo]
        | set[ModuleExportInfo],
        collection3: java.util.Collection[ModuleOpenInfo]
        | typing.Sequence[ModuleOpenInfo]
        | set[ModuleOpenInfo],
        collection4: java.util.Collection[java.lang.classfile.constantpool.ClassEntry]
        | typing.Sequence[java.lang.classfile.constantpool.ClassEntry]
        | set[java.lang.classfile.constantpool.ClassEntry],
        collection5: java.util.Collection[ModuleProvideInfo]
        | typing.Sequence[ModuleProvideInfo]
        | set[ModuleProvideInfo],
    ) -> ModuleAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
        consumer: java.util.function.Consumer[ModuleAttribute.ModuleAttributeBuilder]
        | typing.Callable[[ModuleAttribute.ModuleAttributeBuilder], None],
    ) -> ModuleAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        moduleDesc: java.lang.constant.ModuleDesc,
        consumer: java.util.function.Consumer[ModuleAttribute.ModuleAttributeBuilder]
        | typing.Callable[[ModuleAttribute.ModuleAttributeBuilder], None],
    ) -> ModuleAttribute: ...
    def opens(self) -> java.util.List[ModuleOpenInfo]: ...
    def provides(self) -> java.util.List[ModuleProvideInfo]: ...
    def requires(self) -> java.util.List[ModuleRequireInfo]: ...
    def uses(self) -> java.util.List[java.lang.classfile.constantpool.ClassEntry]: ...
    class ModuleAttributeBuilder:
        @typing.overload
        def exports(
            self, moduleExportInfo: ModuleExportInfo
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def exports(
            self,
            packageDesc: java.lang.constant.PackageDesc,
            int: int,
            *moduleDesc: java.lang.constant.ModuleDesc,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def exports(
            self,
            packageDesc: java.lang.constant.PackageDesc,
            collection: java.util.Collection[java.lang.reflect.AccessFlag]
            | typing.Sequence[java.lang.reflect.AccessFlag]
            | set[java.lang.reflect.AccessFlag],
            *moduleDesc: java.lang.constant.ModuleDesc,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def moduleFlags(self, int: int) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def moduleFlags(
            self, *accessFlag: java.lang.reflect.AccessFlag
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        def moduleName(
            self, moduleDesc: java.lang.constant.ModuleDesc
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        def moduleVersion(
            self, string: java.lang.String | str
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def opens(
            self, moduleOpenInfo: ModuleOpenInfo
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def opens(
            self,
            packageDesc: java.lang.constant.PackageDesc,
            int: int,
            *moduleDesc: java.lang.constant.ModuleDesc,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def opens(
            self,
            packageDesc: java.lang.constant.PackageDesc,
            collection: java.util.Collection[java.lang.reflect.AccessFlag]
            | typing.Sequence[java.lang.reflect.AccessFlag]
            | set[java.lang.reflect.AccessFlag],
            *moduleDesc: java.lang.constant.ModuleDesc,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def provides(
            self, moduleProvideInfo: ModuleProvideInfo
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def provides(
            self,
            classDesc: java.lang.constant.ClassDesc,
            *classDesc2: java.lang.constant.ClassDesc,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def requires(
            self, moduleRequireInfo: ModuleRequireInfo
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def requires(
            self,
            moduleDesc: java.lang.constant.ModuleDesc,
            int: int,
            string: java.lang.String | str,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def requires(
            self,
            moduleDesc: java.lang.constant.ModuleDesc,
            collection: java.util.Collection[java.lang.reflect.AccessFlag]
            | typing.Sequence[java.lang.reflect.AccessFlag]
            | set[java.lang.reflect.AccessFlag],
            string: java.lang.String | str,
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def uses(
            self, classEntry: java.lang.classfile.constantpool.ClassEntry
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...
        @typing.overload
        def uses(
            self, classDesc: java.lang.constant.ClassDesc
        ) -> ModuleAttribute.ModuleAttributeBuilder: ...

class ModuleExportInfo:
    def exportedPackage(self) -> java.lang.classfile.constantpool.PackageEntry: ...
    def exportsFlags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def exportsFlagsMask(self) -> int: ...
    def exportsTo(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ModuleEntry]: ...
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        int: int,
        *moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        int: int,
        list: java.util.List[java.lang.classfile.constantpool.ModuleEntry],
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        *moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        list: java.util.List[java.lang.classfile.constantpool.ModuleEntry],
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        int: int,
        *moduleDesc: java.lang.constant.ModuleDesc,
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        int: int,
        list: java.util.List[java.lang.constant.ModuleDesc],
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        *moduleDesc: java.lang.constant.ModuleDesc,
    ) -> ModuleExportInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        list: java.util.List[java.lang.constant.ModuleDesc],
    ) -> ModuleExportInfo: ...

class ModuleHashInfo:
    def hash(self) -> typing.MutableSequence[int]: ...
    def moduleName(self) -> java.lang.classfile.constantpool.ModuleEntry: ...
    @typing.overload
    @staticmethod
    def of(
        moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> ModuleHashInfo: ...
    @typing.overload
    @staticmethod
    def of(
        moduleDesc: java.lang.constant.ModuleDesc,
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> ModuleHashInfo: ...

class ModuleHashesAttribute(
    java.lang.classfile.Attribute["ModuleHashesAttribute"],
    java.lang.classfile.ClassElement,
):
    def algorithm(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def hashes(self) -> java.util.List[ModuleHashInfo]: ...
    @typing.overload
    @staticmethod
    def of(
        string: java.lang.String | str, *moduleHashInfo: ModuleHashInfo
    ) -> ModuleHashesAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        string: java.lang.String | str, list: java.util.List[ModuleHashInfo]
    ) -> ModuleHashesAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
        *moduleHashInfo: ModuleHashInfo,
    ) -> ModuleHashesAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
        list: java.util.List[ModuleHashInfo],
    ) -> ModuleHashesAttribute: ...

class ModuleMainClassAttribute(
    java.lang.classfile.Attribute["ModuleMainClassAttribute"],
    java.lang.classfile.ClassElement,
):
    def mainClass(self) -> java.lang.classfile.constantpool.ClassEntry: ...
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
    ) -> ModuleMainClassAttribute: ...
    @typing.overload
    @staticmethod
    def of(classDesc: java.lang.constant.ClassDesc) -> ModuleMainClassAttribute: ...

class ModuleOpenInfo:
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        int: int,
        *moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        int: int,
        list: java.util.List[java.lang.classfile.constantpool.ModuleEntry],
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        *moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageEntry: java.lang.classfile.constantpool.PackageEntry,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        list: java.util.List[java.lang.classfile.constantpool.ModuleEntry],
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        int: int,
        *moduleDesc: java.lang.constant.ModuleDesc,
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        int: int,
        list: java.util.List[java.lang.constant.ModuleDesc],
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        *moduleDesc: java.lang.constant.ModuleDesc,
    ) -> ModuleOpenInfo: ...
    @typing.overload
    @staticmethod
    def of(
        packageDesc: java.lang.constant.PackageDesc,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        list: java.util.List[java.lang.constant.ModuleDesc],
    ) -> ModuleOpenInfo: ...
    def openedPackage(self) -> java.lang.classfile.constantpool.PackageEntry: ...
    def opensFlags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def opensFlagsMask(self) -> int: ...
    def opensTo(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ModuleEntry]: ...

class ModulePackagesAttribute(
    java.lang.classfile.Attribute["ModulePackagesAttribute"],
    java.lang.classfile.ClassElement,
):
    @typing.overload
    @staticmethod
    def of(
        *packageEntry: java.lang.classfile.constantpool.PackageEntry,
    ) -> ModulePackagesAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.constantpool.PackageEntry],
    ) -> ModulePackagesAttribute: ...
    @typing.overload
    @staticmethod
    def ofNames(
        *packageDesc: java.lang.constant.PackageDesc,
    ) -> ModulePackagesAttribute: ...
    @typing.overload
    @staticmethod
    def ofNames(
        list: java.util.List[java.lang.constant.PackageDesc],
    ) -> ModulePackagesAttribute: ...
    def packages(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.PackageEntry]: ...

class ModuleProvideInfo:
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
        *classEntry2: java.lang.classfile.constantpool.ClassEntry,
    ) -> ModuleProvideInfo: ...
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
        list: java.util.List[java.lang.classfile.constantpool.ClassEntry],
    ) -> ModuleProvideInfo: ...
    @typing.overload
    @staticmethod
    def of(
        classDesc: java.lang.constant.ClassDesc,
        *classDesc2: java.lang.constant.ClassDesc,
    ) -> ModuleProvideInfo: ...
    @typing.overload
    @staticmethod
    def of(
        classDesc: java.lang.constant.ClassDesc,
        list: java.util.List[java.lang.constant.ClassDesc],
    ) -> ModuleProvideInfo: ...
    def provides(self) -> java.lang.classfile.constantpool.ClassEntry: ...
    def providesWith(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ClassEntry]: ...

class ModuleRequireInfo:
    def has(self, accessFlag: java.lang.reflect.AccessFlag) -> bool: ...
    @typing.overload
    @staticmethod
    def of(
        moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
        int: int,
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> ModuleRequireInfo: ...
    @typing.overload
    @staticmethod
    def of(
        moduleEntry: java.lang.classfile.constantpool.ModuleEntry,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> ModuleRequireInfo: ...
    @typing.overload
    @staticmethod
    def of(
        moduleDesc: java.lang.constant.ModuleDesc,
        int: int,
        string: java.lang.String | str,
    ) -> ModuleRequireInfo: ...
    @typing.overload
    @staticmethod
    def of(
        moduleDesc: java.lang.constant.ModuleDesc,
        collection: java.util.Collection[java.lang.reflect.AccessFlag]
        | typing.Sequence[java.lang.reflect.AccessFlag]
        | set[java.lang.reflect.AccessFlag],
        string: java.lang.String | str,
    ) -> ModuleRequireInfo: ...
    def requires(self) -> java.lang.classfile.constantpool.ModuleEntry: ...
    def requiresFlags(self) -> java.util.Set[java.lang.reflect.AccessFlag]: ...
    def requiresFlagsMask(self) -> int: ...
    def requiresVersion(
        self,
    ) -> java.util.Optional[java.lang.classfile.constantpool.Utf8Entry]: ...

class ModuleResolutionAttribute(
    java.lang.classfile.Attribute["ModuleResolutionAttribute"],
    java.lang.classfile.ClassElement,
):
    @staticmethod
    def of(int: int) -> ModuleResolutionAttribute: ...
    def resolutionFlags(self) -> int: ...

class ModuleTargetAttribute(
    java.lang.classfile.Attribute["ModuleTargetAttribute"],
    java.lang.classfile.ClassElement,
):
    @typing.overload
    @staticmethod
    def of(string: java.lang.String | str) -> ModuleTargetAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> ModuleTargetAttribute: ...
    def targetPlatform(self) -> java.lang.classfile.constantpool.Utf8Entry: ...

class NestHostAttribute(
    java.lang.classfile.Attribute["NestHostAttribute"], java.lang.classfile.ClassElement
):
    def nestHost(self) -> java.lang.classfile.constantpool.ClassEntry: ...
    @typing.overload
    @staticmethod
    def of(
        classEntry: java.lang.classfile.constantpool.ClassEntry,
    ) -> NestHostAttribute: ...
    @typing.overload
    @staticmethod
    def of(classDesc: java.lang.constant.ClassDesc) -> NestHostAttribute: ...

class NestMembersAttribute(
    java.lang.classfile.Attribute["NestMembersAttribute"],
    java.lang.classfile.ClassElement,
):
    def nestMembers(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ClassEntry]: ...
    @typing.overload
    @staticmethod
    def of(
        *classEntry: java.lang.classfile.constantpool.ClassEntry,
    ) -> NestMembersAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.constantpool.ClassEntry],
    ) -> NestMembersAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(*classDesc: java.lang.constant.ClassDesc) -> NestMembersAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(
        list: java.util.List[java.lang.constant.ClassDesc],
    ) -> NestMembersAttribute: ...

class PermittedSubclassesAttribute(
    java.lang.classfile.Attribute["PermittedSubclassesAttribute"],
    java.lang.classfile.ClassElement,
):
    @typing.overload
    @staticmethod
    def of(
        *classEntry: java.lang.classfile.constantpool.ClassEntry,
    ) -> PermittedSubclassesAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.constantpool.ClassEntry],
    ) -> PermittedSubclassesAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(
        *classDesc: java.lang.constant.ClassDesc,
    ) -> PermittedSubclassesAttribute: ...
    @typing.overload
    @staticmethod
    def ofSymbols(
        list: java.util.List[java.lang.constant.ClassDesc],
    ) -> PermittedSubclassesAttribute: ...
    def permittedSubclasses(
        self,
    ) -> java.util.List[java.lang.classfile.constantpool.ClassEntry]: ...

class RecordAttribute(
    java.lang.classfile.Attribute["RecordAttribute"], java.lang.classfile.ClassElement
):
    def components(self) -> java.util.List[RecordComponentInfo]: ...
    @typing.overload
    @staticmethod
    def of(*recordComponentInfo: RecordComponentInfo) -> RecordAttribute: ...
    @typing.overload
    @staticmethod
    def of(list: java.util.List[RecordComponentInfo]) -> RecordAttribute: ...

class RecordComponentInfo(java.lang.classfile.AttributedElement):
    def descriptor(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    def descriptorSymbol(self) -> java.lang.constant.ClassDesc: ...
    def name(self) -> java.lang.classfile.constantpool.Utf8Entry: ...
    @typing.overload
    @staticmethod
    def of(
        string: java.lang.String | str,
        classDesc: java.lang.constant.ClassDesc,
        *attribute: java.lang.classfile.Attribute[typing.Any],
    ) -> RecordComponentInfo: ...
    @typing.overload
    @staticmethod
    def of(
        string: java.lang.String | str,
        classDesc: java.lang.constant.ClassDesc,
        list: java.util.List[java.lang.classfile.Attribute[typing.Any]],
    ) -> RecordComponentInfo: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
        utf8Entry2: java.lang.classfile.constantpool.Utf8Entry,
        *attribute: java.lang.classfile.Attribute[typing.Any],
    ) -> RecordComponentInfo: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
        utf8Entry2: java.lang.classfile.constantpool.Utf8Entry,
        list: java.util.List[java.lang.classfile.Attribute[typing.Any]],
    ) -> RecordComponentInfo: ...

class RuntimeInvisibleAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeInvisibleAnnotationsAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
):
    def annotations(self) -> java.util.List[java.lang.classfile.Annotation]: ...
    @typing.overload
    @staticmethod
    def of(
        *annotation: java.lang.classfile.Annotation,
    ) -> RuntimeInvisibleAnnotationsAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.Annotation],
    ) -> RuntimeInvisibleAnnotationsAttribute: ...

class RuntimeInvisibleParameterAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeInvisibleParameterAnnotationsAttribute"],
    java.lang.classfile.MethodElement,
):
    @staticmethod
    def of(
        list: java.util.List[java.util.List[java.lang.classfile.Annotation]],
    ) -> RuntimeInvisibleParameterAnnotationsAttribute: ...
    def parameterAnnotations(
        self,
    ) -> java.util.List[java.util.List[java.lang.classfile.Annotation]]: ...

class RuntimeInvisibleTypeAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeInvisibleTypeAnnotationsAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
    java.lang.classfile.CodeElement,
):
    def annotations(self) -> java.util.List[java.lang.classfile.TypeAnnotation]: ...
    @typing.overload
    @staticmethod
    def of(
        *typeAnnotation: java.lang.classfile.TypeAnnotation,
    ) -> RuntimeInvisibleTypeAnnotationsAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.TypeAnnotation],
    ) -> RuntimeInvisibleTypeAnnotationsAttribute: ...

class RuntimeVisibleAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeVisibleAnnotationsAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
):
    def annotations(self) -> java.util.List[java.lang.classfile.Annotation]: ...
    @typing.overload
    @staticmethod
    def of(
        *annotation: java.lang.classfile.Annotation,
    ) -> RuntimeVisibleAnnotationsAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.Annotation],
    ) -> RuntimeVisibleAnnotationsAttribute: ...

class RuntimeVisibleParameterAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeVisibleParameterAnnotationsAttribute"],
    java.lang.classfile.MethodElement,
):
    @staticmethod
    def of(
        list: java.util.List[java.util.List[java.lang.classfile.Annotation]],
    ) -> RuntimeVisibleParameterAnnotationsAttribute: ...
    def parameterAnnotations(
        self,
    ) -> java.util.List[java.util.List[java.lang.classfile.Annotation]]: ...

class RuntimeVisibleTypeAnnotationsAttribute(
    java.lang.classfile.Attribute["RuntimeVisibleTypeAnnotationsAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
    java.lang.classfile.CodeElement,
):
    def annotations(self) -> java.util.List[java.lang.classfile.TypeAnnotation]: ...
    @typing.overload
    @staticmethod
    def of(
        *typeAnnotation: java.lang.classfile.TypeAnnotation,
    ) -> RuntimeVisibleTypeAnnotationsAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        list: java.util.List[java.lang.classfile.TypeAnnotation],
    ) -> RuntimeVisibleTypeAnnotationsAttribute: ...

class SignatureAttribute(
    java.lang.classfile.Attribute["SignatureAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
):
    def asClassSignature(self) -> java.lang.classfile.ClassSignature: ...
    def asMethodSignature(self) -> java.lang.classfile.MethodSignature: ...
    def asTypeSignature(self) -> java.lang.classfile.Signature: ...
    @typing.overload
    @staticmethod
    def of(
        classSignature: java.lang.classfile.ClassSignature,
    ) -> SignatureAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        methodSignature: java.lang.classfile.MethodSignature,
    ) -> SignatureAttribute: ...
    @typing.overload
    @staticmethod
    def of(signature: java.lang.classfile.Signature) -> SignatureAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> SignatureAttribute: ...
    def signature(self) -> java.lang.classfile.constantpool.Utf8Entry: ...

class SourceDebugExtensionAttribute(
    java.lang.classfile.Attribute["SourceDebugExtensionAttribute"],
    java.lang.classfile.ClassElement,
):
    def contents(self) -> typing.MutableSequence[int]: ...
    @staticmethod
    def of(
        byteArray: list[int] | jpype.JArray | bytes,
    ) -> SourceDebugExtensionAttribute: ...

class SourceFileAttribute(
    java.lang.classfile.Attribute["SourceFileAttribute"],
    java.lang.classfile.ClassElement,
):
    @typing.overload
    @staticmethod
    def of(string: java.lang.String | str) -> SourceFileAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> SourceFileAttribute: ...
    def sourceFile(self) -> java.lang.classfile.constantpool.Utf8Entry: ...

class SourceIDAttribute(
    java.lang.classfile.Attribute["SourceIDAttribute"], java.lang.classfile.ClassElement
):
    @typing.overload
    @staticmethod
    def of(string: java.lang.String | str) -> SourceIDAttribute: ...
    @typing.overload
    @staticmethod
    def of(
        utf8Entry: java.lang.classfile.constantpool.Utf8Entry,
    ) -> SourceIDAttribute: ...
    def sourceId(self) -> java.lang.classfile.constantpool.Utf8Entry: ...

class StackMapTableAttribute(
    java.lang.classfile.Attribute["StackMapTableAttribute"],
    java.lang.classfile.CodeElement,
):
    def entries(self) -> java.util.List[StackMapFrameInfo]: ...
    @staticmethod
    def of(list: java.util.List[StackMapFrameInfo]) -> StackMapTableAttribute: ...

class SyntheticAttribute(
    java.lang.classfile.Attribute["SyntheticAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
):
    @staticmethod
    def of() -> SyntheticAttribute: ...

class UnknownAttribute(
    java.lang.classfile.Attribute["UnknownAttribute"],
    java.lang.classfile.ClassElement,
    java.lang.classfile.MethodElement,
    java.lang.classfile.FieldElement,
    java.lang.classfile.CodeElement,
):
    def contents(self) -> typing.MutableSequence[int]: ...

class StackMapFrameInfo:
    def frameType(self) -> int: ...
    def locals(self) -> java.util.List[StackMapFrameInfo.VerificationTypeInfo]: ...
    @staticmethod
    def of(
        label: java.lang.classfile.Label,
        list: java.util.List[StackMapFrameInfo.VerificationTypeInfo],
        list2: java.util.List[StackMapFrameInfo.VerificationTypeInfo],
    ) -> StackMapFrameInfo: ...
    def stack(self) -> java.util.List[StackMapFrameInfo.VerificationTypeInfo]: ...
    def target(self) -> java.lang.classfile.Label: ...
    class ObjectVerificationTypeInfo(
        java.lang.classfile.attribute.StackMapFrameInfo.VerificationTypeInfo
    ):
        def className(self) -> java.lang.classfile.constantpool.ClassEntry: ...
        def classSymbol(self) -> java.lang.constant.ClassDesc: ...
        @typing.overload
        @staticmethod
        def of(
            classEntry: java.lang.classfile.constantpool.ClassEntry,
        ) -> StackMapFrameInfo.ObjectVerificationTypeInfo: ...
        @typing.overload
        @staticmethod
        def of(
            classDesc: java.lang.constant.ClassDesc,
        ) -> StackMapFrameInfo.ObjectVerificationTypeInfo: ...

    class SimpleVerificationTypeInfo(
        java.lang.Enum["StackMapFrameInfo.SimpleVerificationTypeInfo"],
        java.lang.classfile.attribute.StackMapFrameInfo.VerificationTypeInfo,
    ):
        TOP: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        INTEGER: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        FLOAT: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        DOUBLE: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        LONG: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        NULL: typing.ClassVar[StackMapFrameInfo.SimpleVerificationTypeInfo] = ...
        UNINITIALIZED_THIS: typing.ClassVar[
            StackMapFrameInfo.SimpleVerificationTypeInfo
        ] = ...
        def tag(self) -> int: ...
        _valueOf_0__T = typing.TypeVar("_valueOf_0__T", bound=java.lang.Enum)  # <T>
        @typing.overload
        @staticmethod
        def valueOf(
            class_: type[_valueOf_0__T], string: java.lang.String | str
        ) -> _valueOf_0__T: ...
        @typing.overload
        @staticmethod
        def valueOf(
            string: java.lang.String | str,
        ) -> StackMapFrameInfo.SimpleVerificationTypeInfo: ...
        @staticmethod
        def values() -> typing.MutableSequence[
            StackMapFrameInfo.SimpleVerificationTypeInfo
        ]: ...

    class UninitializedVerificationTypeInfo(
        java.lang.classfile.attribute.StackMapFrameInfo.VerificationTypeInfo
    ):
        def newTarget(self) -> java.lang.classfile.Label: ...
        @staticmethod
        def of(
            label: java.lang.classfile.Label,
        ) -> StackMapFrameInfo.UninitializedVerificationTypeInfo: ...

    class VerificationTypeInfo:
        ITEM_TOP: typing.ClassVar[int] = ...
        ITEM_INTEGER: typing.ClassVar[int] = ...
        ITEM_FLOAT: typing.ClassVar[int] = ...
        ITEM_DOUBLE: typing.ClassVar[int] = ...
        ITEM_LONG: typing.ClassVar[int] = ...
        ITEM_NULL: typing.ClassVar[int] = ...
        ITEM_UNINITIALIZED_THIS: typing.ClassVar[int] = ...
        ITEM_OBJECT: typing.ClassVar[int] = ...
        ITEM_UNINITIALIZED: typing.ClassVar[int] = ...
        def tag(self) -> int: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.lang.classfile.attribute")``.

    AnnotationDefaultAttribute: type[AnnotationDefaultAttribute]
    BootstrapMethodsAttribute: type[BootstrapMethodsAttribute]
    CharacterRangeInfo: type[CharacterRangeInfo]
    CharacterRangeTableAttribute: type[CharacterRangeTableAttribute]
    CodeAttribute: type[CodeAttribute]
    CompilationIDAttribute: type[CompilationIDAttribute]
    ConstantValueAttribute: type[ConstantValueAttribute]
    DeprecatedAttribute: type[DeprecatedAttribute]
    EnclosingMethodAttribute: type[EnclosingMethodAttribute]
    ExceptionsAttribute: type[ExceptionsAttribute]
    InnerClassInfo: type[InnerClassInfo]
    InnerClassesAttribute: type[InnerClassesAttribute]
    LineNumberInfo: type[LineNumberInfo]
    LineNumberTableAttribute: type[LineNumberTableAttribute]
    LocalVariableInfo: type[LocalVariableInfo]
    LocalVariableTableAttribute: type[LocalVariableTableAttribute]
    LocalVariableTypeInfo: type[LocalVariableTypeInfo]
    LocalVariableTypeTableAttribute: type[LocalVariableTypeTableAttribute]
    MethodParameterInfo: type[MethodParameterInfo]
    MethodParametersAttribute: type[MethodParametersAttribute]
    ModuleAttribute: type[ModuleAttribute]
    ModuleExportInfo: type[ModuleExportInfo]
    ModuleHashInfo: type[ModuleHashInfo]
    ModuleHashesAttribute: type[ModuleHashesAttribute]
    ModuleMainClassAttribute: type[ModuleMainClassAttribute]
    ModuleOpenInfo: type[ModuleOpenInfo]
    ModulePackagesAttribute: type[ModulePackagesAttribute]
    ModuleProvideInfo: type[ModuleProvideInfo]
    ModuleRequireInfo: type[ModuleRequireInfo]
    ModuleResolutionAttribute: type[ModuleResolutionAttribute]
    ModuleTargetAttribute: type[ModuleTargetAttribute]
    NestHostAttribute: type[NestHostAttribute]
    NestMembersAttribute: type[NestMembersAttribute]
    PermittedSubclassesAttribute: type[PermittedSubclassesAttribute]
    RecordAttribute: type[RecordAttribute]
    RecordComponentInfo: type[RecordComponentInfo]
    RuntimeInvisibleAnnotationsAttribute: type[RuntimeInvisibleAnnotationsAttribute]
    RuntimeInvisibleParameterAnnotationsAttribute: type[
        RuntimeInvisibleParameterAnnotationsAttribute
    ]
    RuntimeInvisibleTypeAnnotationsAttribute: type[
        RuntimeInvisibleTypeAnnotationsAttribute
    ]
    RuntimeVisibleAnnotationsAttribute: type[RuntimeVisibleAnnotationsAttribute]
    RuntimeVisibleParameterAnnotationsAttribute: type[
        RuntimeVisibleParameterAnnotationsAttribute
    ]
    RuntimeVisibleTypeAnnotationsAttribute: type[RuntimeVisibleTypeAnnotationsAttribute]
    SignatureAttribute: type[SignatureAttribute]
    SourceDebugExtensionAttribute: type[SourceDebugExtensionAttribute]
    SourceFileAttribute: type[SourceFileAttribute]
    SourceIDAttribute: type[SourceIDAttribute]
    StackMapFrameInfo: type[StackMapFrameInfo]
    StackMapTableAttribute: type[StackMapTableAttribute]
    SyntheticAttribute: type[SyntheticAttribute]
    UnknownAttribute: type[UnknownAttribute]
