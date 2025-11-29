from __future__ import annotations

import logging
import os
import warnings
from contextlib import suppress
from functools import cache
from typing import Any

import jpype
import numpy as np
import scyjava
import scyjava.config

MAVEN_COORDINATE = "ome:formats-gpl:RELEASE"

# Configure Java constraints from environment variables
# BFF_JAVA_VENDOR: Java vendor (e.g., "zulu-jre", "adoptium", "temurin")
# BFF_JAVA_VERSION: Java version (e.g., "11", "17", "21")
_bff_vendor = os.getenv("BFF_JAVA_VENDOR") or None
_bff_version = os.getenv("BFF_JAVA_VERSION") or None
if _bff_vendor or _bff_version:
    _kwargs = {}
    if _bff_vendor:
        _kwargs["vendor"] = _bff_vendor
    if _bff_version:
        _kwargs["version"] = _bff_version
    # Force cjdk to always be used when constraints are set, even if system Java exists
    _kwargs["fetch"] = "always"
    scyjava.config.set_java_constraints(**_kwargs)

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

_LOGGING_REDIRECTED = False


def redirect_java_logging(logger: logging.Logger | None = None) -> None:
    """Redirect Java logging to Python logger."""
    global _LOGGING_REDIRECTED
    if _LOGGING_REDIRECTED:
        return

    scyjava.start_jvm()  # won't start if already running

    _logger = logger or LOGGER

    class PyAppender:
        def doAppend(self, event: Any) -> None:
            # event is an ILoggingEvent
            msg = str(event.getFormattedMessage())
            level = str(event.getLevel())
            # dispatch to Python logger
            getattr(logger, level.lower(), _logger.info)(msg)

    # Create a proxy for the Appender interface
    proxy = jpype.JProxy("ch.qos.logback.core.Appender", inst=PyAppender())

    # Get the LoggerContext
    Slf4jFactory = scyjava.jimport("org.slf4j.LoggerFactory")
    root = Slf4jFactory.getILoggerFactory().getLogger("ROOT")

    # remove the console appender
    for appender in root.iteratorForAppenders():
        if appender.getName() == "console":
            root.detachAppender(appender)
            break

    # add the Python appender
    root.addAppender(proxy)
    _LOGGING_REDIRECTED = True


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
