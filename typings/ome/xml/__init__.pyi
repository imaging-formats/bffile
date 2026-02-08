from typing import Protocol

import ome.xml.meta
import ome.xml.model

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("ome.xml")``.

    meta: ome.xml.meta.__module_protocol__
    model: ome.xml.model.__module_protocol__
