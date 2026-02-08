from typing import Protocol

import java.lang
import loci.poi.ddf
import loci.poi.dev
import loci.poi.hpsf
import loci.poi.hssf
import loci.poi.poifs
import loci.poi.util

class POIDocument:
    def __init__(self): ...
    def getDocumentSummaryInformation(
        self,
    ) -> loci.poi.hpsf.DocumentSummaryInformation: ...
    def getSummaryInformation(self) -> loci.poi.hpsf.SummaryInformation: ...

class POITextExtractor:
    def __init__(self, pOIDocument: POIDocument): ...
    def getText(self) -> java.lang.String: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi")``.

    POIDocument: type[POIDocument]
    POITextExtractor: type[POITextExtractor]
    ddf: loci.poi.ddf.__module_protocol__
    dev: loci.poi.dev.__module_protocol__
    hpsf: loci.poi.hpsf.__module_protocol__
    hssf: loci.poi.hssf.__module_protocol__
    poifs: loci.poi.poifs.__module_protocol__
    util: loci.poi.util.__module_protocol__
