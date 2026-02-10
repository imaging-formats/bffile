"""Yet another Bio-Formats wrapper for Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bffile")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._biofile import BioFile
from ._imread import imread
from ._lazy_array import LazyBioArray
from ._series import Series

__all__ = ["BioFile", "LazyBioArray", "Series", "imread"]
