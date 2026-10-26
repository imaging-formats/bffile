"""Java (logback) log events must reach Python at their own level."""

from __future__ import annotations

import logging

import pytest

from bffile._jimports import jimport


@pytest.mark.parametrize(
    ("method", "level"),
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warn", logging.WARNING),
        ("error", logging.ERROR),
    ],
)
def test_java_log_level(
    method: str, level: int, caplog: pytest.LogCaptureFixture
) -> None:
    java_logger = jimport("org.slf4j.LoggerFactory").getLogger("bffile.test")
    msg = f"java {method} message"
    with caplog.at_level(logging.DEBUG, logger="bffile"):
        getattr(java_logger, method)(msg)
    assert [(r.levelno, r.message) for r in caplog.records if r.message == msg] == [
        (level, msg)
    ]


def test_java_warning_visible_at_default_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Python's default level is WARNING: Java warnings must not be dropped
    java_logger = jimport("org.slf4j.LoggerFactory").getLogger("bffile.test")
    with caplog.at_level(logging.WARNING, logger="bffile"):
        java_logger.debug("java debug noise")
        java_logger.warn("java warning")
    assert [r.message for r in caplog.records] == ["java warning"]
