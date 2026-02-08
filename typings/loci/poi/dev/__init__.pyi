from typing import Protocol

import java.lang
import jpype

class RecordGenerator:
    def __init__(self): ...
    @staticmethod
    def main(stringArray: list[java.lang.String] | jpype.JArray) -> None: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.dev")``.

    RecordGenerator: type[RecordGenerator]
