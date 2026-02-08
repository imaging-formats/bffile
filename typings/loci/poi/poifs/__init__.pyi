from typing import Protocol

import loci.poi.poifs.common
import loci.poi.poifs.dev
import loci.poi.poifs.eventfilesystem
import loci.poi.poifs.filesystem
import loci.poi.poifs.property
import loci.poi.poifs.storage

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("loci.poi.poifs")``.

    common: loci.poi.poifs.common.__module_protocol__
    dev: loci.poi.poifs.dev.__module_protocol__
    eventfilesystem: loci.poi.poifs.eventfilesystem.__module_protocol__
    filesystem: loci.poi.poifs.filesystem.__module_protocol__
    property: loci.poi.poifs.property.__module_protocol__
    storage: loci.poi.poifs.storage.__module_protocol__
