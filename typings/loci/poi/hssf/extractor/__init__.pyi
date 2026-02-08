import typing
from typing import Protocol

import java.lang
import loci.poi
import loci.poi.hssf.usermodel
import loci.poi.poifs.filesystem

class ExcelExtractor(loci.poi.POITextExtractor):
    @typing.overload
    def __init__(self, hSSFWorkbook: loci.poi.hssf.usermodel.HSSFWorkbook): ...
    @typing.overload
    def __init__(self, pOIFSFileSystem: loci.poi.poifs.filesystem.POIFSFileSystem): ...
    def getText(self) -> java.lang.String: ...
    def setFormulasNotResults(self, boolean: bool) -> None: ...
    def setIncludeSheetNames(self, boolean: bool) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.hssf.extractor")``.

    ExcelExtractor: type[ExcelExtractor]
