from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import warnings
from collections.abc import Mapping, Sequence, Set
from functools import cache
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, Literal, overload
from unittest.mock import patch

import jpype
import numpy as np
import scyjava
import scyjava.config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


MAVEN_COORDINATE = "ome:formats-gpl:RELEASE"

# Check if the BIOFORMATS_VERSION environment variable is set
# and if so, use it as the Maven coordinate
if coord := os.getenv("BIOFORMATS_VERSION", ""):
    # allow a single version number to be passed
    if ":" not in coord and all(x.isdigit() for x in coord.split(".")):
        # if the coordinate is just a version number, use the default group and artifact
        coord = f"ome:formats-gpl:{coord}"

    # ensure the coordinate is valid
    if 2 > len(coord.split(":")) > 5:
        warnings.warn(
            f"Invalid BIOFORMATS_VERSION env var: {coord!r}. "
            "Must be a valid maven coordinate with 2-5 elements. "
            f"Using default {MAVEN_COORDINATE!r}",
            stacklevel=2,
        )
    else:
        MAVEN_COORDINATE = coord

scyjava.config.endpoints.append(MAVEN_COORDINATE)
# NB: logback 1.3.x is the last version with Java 8 support!
scyjava.config.endpoints.append("ch.qos.logback:logback-classic:1.3.15")

# #################################### LOGGING ####################################

# python-side logger

LOGGER = logging.getLogger("bffile")
fmt = (
    "%(asctime)s.%(msecs)03d "  # timestamp with milliseconds
    "[%(levelname)-5s] "  # level, padded
    "%(name)s:%(lineno)d - "  # logger name and line no.
    "%(message)s"  # the log message
)
datefmt = "%Y-%m-%d %H:%M:%S"
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
# avoid double-logs if somebody has already attached handlers
if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
    LOGGER.addHandler(handler)


def redirect_java_logging(logger: logging.Logger | None = None) -> None:
    """Redirect Java logging to Python logger."""
    _logger = logger or LOGGER

    class PyAppender:
        def doAppend(self, event: Any) -> None:
            # event is an ILoggingEvent
            msg = str(event.getFormattedMessage())
            level = str(event.getLevel())
            # dispatch to Python logger
            getattr(logger, level.lower(), _logger.info)(msg)

        def getName(self) -> str:
            return "PyAppender"

    # Create a proxy for the Appender interface
    proxy = jpype.JProxy("ch.qos.logback.core.Appender", inst=PyAppender())

    # Get the LoggerContext
    Slf4jFactory = scyjava.jimport("org.slf4j.LoggerFactory")
    root = Slf4jFactory.getILoggerFactory().getLogger("ROOT")

    # remove the console appender
    with contextlib.suppress(AttributeError):
        for appender in root.iteratorForAppenders():
            if appender.getName() in ("console", "PyAppender"):
                root.detachAppender(appender)

        # add the Python appender
        root.addAppender(proxy)


def pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a loci.formats PixelType integer into a numpy dtype."""
    FormatTools = scyjava.jimport("loci.formats.FormatTools")

    fmt2type: dict[int, str] = {
        FormatTools.INT8: "i1",
        FormatTools.UINT8: "u1",
        FormatTools.INT16: "i2",
        FormatTools.UINT16: "u2",
        FormatTools.INT32: "i4",
        FormatTools.UINT32: "u4",
        FormatTools.FLOAT: "f4",
        FormatTools.DOUBLE: "f8",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


@cache
def hide_memoization_warning() -> None:
    """HACK: this silences a warning about memoization for now.

    An illegal reflective access operation has occurred
    https://github.com/ome/bioformats/issues/3659
    """
    with contextlib.suppress(Exception):
        import jpype

        System = jpype.JPackage("java").lang.System
        System.err.close()


@contextlib.contextmanager
def path_prepended(path: Path | str) -> Iterator[None]:
    """
    Context manager to temporarily prepend the given path to PATH.
    """
    save_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([str(path), save_path])
    try:
        yield
    finally:
        os.environ["PATH"] = save_path


@cache
def start_jvm() -> None:
    """Start the JVM if not already running."""
    with _setup_java():
        scyjava.start_jvm()  # won't repeat if already running
        redirect_java_logging()
        hide_memoization_warning()


subprocess_check_output = subprocess.check_output


def _silent_check_output(*args: Any, **kwargs: Any) -> Any:
    # also suppress stderr on calls to subprocess.check_output
    kwargs.setdefault("stderr", subprocess.DEVNULL)
    return subprocess_check_output(*args, **kwargs)


if MAVEN_URL := os.getenv("MAVEN_URL", ""):
    MAVEN_SHA = os.getenv("MAVEN_SHA", "")
    if MAVEN_URL.endswith("tar.gz") and MAVEN_URL.startswith("http"):
        MAVEN_URL = MAVEN_URL.replace("http", "tgz+http")
else:
    MAVEN_URL = "tgz+https://dlcdn.apache.org/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz"
    MAVEN_SHA = "a555254d6b53d267965a3404ecb14e53c3827c09c3b94b5678835887ab404556bfaf78dcfe03ba76fa2508649dca8531c74bca4d5846513522404d48e8c4ac8b"  # noqa: E501


@contextlib.contextmanager
def _setup_java() -> Iterator[str]:
    """Ensure we have a capable JVM and maven on the path."""
    try:
        with patch.object(subprocess, "check_output", new=_silent_check_output):
            jpype.getDefaultJVMPath()
            java_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
    # on Darwin, may raise a CalledProcessError on `/user/libexec/java_home`
    except (jpype.JVMNotFoundException, CalledProcessError):
        try:
            import cjdk
        except ImportError:
            raise ImportError(
                "No JVM found and cjdk is not installed (so we can't install one for "
                "you).\n\nPlease either install java manually "
                "(e.g. https://adoptium.net/installation/),\nor `pip install cjdk` to "
                "allow us to install one automatically for you."
            ) from None

        LOGGER.info("No JVM found, installing Zulu JRE 11 via cjdk...")
        vendor = os.getenv("JAVA_VENDOR", "zulu-jre")
        version = os.getenv("JAVA_VERSION", "11")
        java_ctx = cjdk.java_env(vendor=vendor, version=version)

    with java_ctx:
        if not shutil.which("mvn"):
            try:
                import cjdk
            except ImportError:
                raise ImportError(
                    "Maven not found on PATH and cjdk is not installed.\n\n"
                    "Maven is required to fetch the Bio-Formats JAR and other java "
                    "dependencies.\nPlease either install maven manually, or "
                    "`pip install cjdk` to allow us to install it for you."
                ) from None

            # determine sha length based on MAVEN_SHA
            kwargs = {}
            if sha_len := len(MAVEN_SHA):
                # assuming hex-encoded SHA, length should be 40, 64, or 128
                sha_lengths = {40: "sha1", 64: "sha256", 128: "sha512"}
                if sha_len not in sha_lengths:
                    raise ValueError(
                        "MAVEN_SHA be a valid sha1, sha256, or sha512 hash."
                        f"Got invalid SHA length: {sha_len}. "
                    )
                kwargs = {sha_lengths[sha_len]: MAVEN_SHA}

            maven_dir = cjdk.cache_package("Maven", MAVEN_URL, **kwargs)
            if _mvn := next(maven_dir.rglob("apache-maven-*/**/mvn"), None):
                with path_prepended(_mvn.parent):
                    scyjava.start_jvm()

        yield jpype.getDefaultJVMPath()


if TYPE_CHECKING:
    from loci.common.services import ServiceFactory
    from loci.formats import FormatTools, ImageReader, Memoizer
    from loci.formats.in_ import DynamicMetadataOptions
    from loci.formats.ome import OMEPyramidStore
    from loci.formats.services import OMEXMLService


@overload
def jimport(
    classname: Literal["loci.formats.ome.OMEPyramidStore"],
) -> type[OMEPyramidStore]: ...
@overload
def jimport(
    classname: Literal["loci.formats.in_.DynamicMetadataOptions"],
) -> type[DynamicMetadataOptions]: ...
@overload
def jimport(
    classname: Literal["loci.formats.FormatTools"],
) -> type[FormatTools]: ...
@overload
def jimport(
    classname: Literal["loci.formats.services.OMEXMLService"],
) -> type[OMEXMLService]: ...
@overload
def jimport(
    classname: Literal["loci.common.services.ServiceFactory"],
) -> type[ServiceFactory]: ...
@overload
def jimport(classname: Literal["loci.formats.ImageReader"]) -> type[ImageReader]: ...
@overload
def jimport(classname: Literal["loci.formats.Memoizer"]) -> type[Memoizer]: ...
@overload
def jimport(classname: str) -> Any: ...
def jimport(classname: str) -> Any:
    return scyjava.jimport(classname)


@jpype.JImplementationFor("java.util.Map")
class _Map(Mapping):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"


@jpype.JImplementationFor("java.util.List")
class _ArrayList(Sequence):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"


@jpype.JImplementationFor("java.util.Set")
class _Set(Set):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({set(self)})"


@jpype.JImplementationFor("loci.formats.CoreMetadata")
class _CoreMetadata:
    def __repr__(self) -> str:
        return f"\n{self}\n"
