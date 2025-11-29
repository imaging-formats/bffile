"""Test that BFF_JAVA_VERSION and BFF_JAVA_VENDOR environment variables work."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SCRIPT = """\
import bffile._java_stuff  # noqa: F401
import scyjava
scyjava.start_jvm()
System = scyjava.jimport("java.lang.System")
print(System.getProperty("java.version"))
"""


@pytest.mark.parametrize(
    "vendor,version",
    [("", "17"), ("", "21"), ("adoptium", "17"), ("temurin", "21"), ("zulu-jre", "11")],
)
def test_bff_java_constraints(vendor: str, version: str) -> None:
    """Test BFF_JAVA_VENDOR and BFF_JAVA_VERSION environment variables."""
    env = os.environ.copy()

    # Remove BFF and Java-related variables
    env.pop("BFF_JAVA_VENDOR", None)
    env.pop("BFF_JAVA_VERSION", None)
    env.pop("JAVA_HOME", None)

    # Clean PATH
    path_parts = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join(
        p for p in path_parts if "cjdk" not in p.lower() and "/java" not in p.lower()
    )

    # Set requested BFF variables
    if vendor:
        env["BFF_JAVA_VENDOR"] = vendor
    if version:
        env["BFF_JAVA_VERSION"] = version

    result = subprocess.check_output([sys.executable, "-c", SCRIPT], env=env)

    java_version = result.decode().strip().split("\n")[-1]
    assert java_version.startswith(f"{version}."), (
        f"Expected Java {version}, got {java_version}"
    )
