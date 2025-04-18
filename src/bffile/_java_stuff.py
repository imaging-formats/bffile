from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, suppress
from functools import cache
from typing import TYPE_CHECKING, Any

import jgo
import jpype
import numpy as np
import scyjava
import scyjava.config

if TYPE_CHECKING:
    from pathlib import Path

MAVEN_COORDINATE = "ome:formats-gpl:RELEASE"

# Check if the BIOFORMATS_VERSION environment variable is set
# and if so, use it as the Maven coordinate
if (coord := os.getenv("BIOFORMATS_VERSION", None)) is not None:
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
    with suppress(Exception):
        import jpype

        System = jpype.JPackage("java").lang.System
        System.err.close()


maven_url = "tgz+https://dlcdn.apache.org/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz"
maven_sha512 = "a555254d6b53d267965a3404ecb14e53c3827c09c3b94b5678835887ab404556bfaf78dcfe03ba76fa2508649dca8531c74bca4d5846513522404d48e8c4ac8b"


@contextmanager
def path_prepended(path: Path | str) -> None:
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
    try:
        scyjava.start_jvm()  # won't repeat if already running
    except jgo.jgo.ExecutableNotFound as e:
        executable = e.args[0].split(" ")[0]
        if executable == "mvn":
            import cjdk

            maven_dir = cjdk.cache_package("Maven", maven_url, sha512=maven_sha512)
            if mvn := next(maven_dir.rglob("apache-maven-*/**/mvn"), None):
                with path_prepended(mvn.parent):
                    scyjava.start_jvm()

    # redirect_java_logging()
    # hide_memoization_warning()
