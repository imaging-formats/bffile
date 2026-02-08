from __future__ import annotations

import contextlib
import logging
import os
import warnings
from functools import cache
from typing import Any

import jpype
import scyjava
import scyjava.config

MAVEN_COORDINATE = "ome:formats-gpl:RELEASE"

# Configure Java constraints from environment variables
# BFF_JAVA_VENDOR: Java vendor (e.g., "zulu-jre", "adoptium", "temurin")
# BFF_JAVA_VERSION: Java version (e.g., "11", "17", "21")
# BFF_JAVA_FETCH: Fetch mode ("always", "prefer", "never", default is auto)
_bff_vendor = os.getenv("BFF_JAVA_VENDOR") or None
_bff_version = os.getenv("BFF_JAVA_VERSION") or None
_bff_fetch = os.getenv("BFF_JAVA_FETCH") or None
if _bff_vendor or _bff_version:
    _kwargs = {}
    if _bff_vendor:
        _kwargs["vendor"] = _bff_vendor
    if _bff_version:
        _kwargs["version"] = _bff_version
    # Control fetch behavior via environment variable
    # Default: don't force download unless explicitly requested
    if _bff_fetch:
        _kwargs["fetch"] = _bff_fetch
    scyjava.config.set_java_constraints(**_kwargs)

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


@cache  # run only once
def start_jvm() -> None:
    """Start the JVM if not already running."""
    scyjava.start_jvm()  # won't repeat if already running
    redirect_java_logging()
    hide_memoization_warning()
