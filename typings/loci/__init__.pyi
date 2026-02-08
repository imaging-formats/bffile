from typing import Protocol

import loci.common
import loci.formats
import loci.poi

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci")``.

    common: loci.common.__module_protocol__
    formats: loci.formats.__module_protocol__
    poi: loci.poi.__module_protocol__
