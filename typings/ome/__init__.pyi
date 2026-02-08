from typing import Protocol

import ome.codecs
import ome.jxrlib
import ome.metakit
import ome.specification
import ome.units
import ome.xml

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome")``.

    codecs: ome.codecs.__module_protocol__
    jxrlib: ome.jxrlib.__module_protocol__
    metakit: ome.metakit.__module_protocol__
    specification: ome.specification.__module_protocol__
    units: ome.units.__module_protocol__
    xml: ome.xml.__module_protocol__
