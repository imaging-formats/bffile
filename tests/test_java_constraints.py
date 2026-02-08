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
print(f"VERSION:{System.getProperty('java.version')}")
print(f"JAVA_HOME:{System.getProperty('java.home')}")
print(f"VENDOR:{System.getProperty('java.vendor')}")
"""


def test_java_constraints_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test: verify env vars are correctly passed to scyjava.

    This is a fast unit test that mocks scyjava to verify that bffile
    correctly reads BFF_JAVA_* env vars and passes them to scyjava.
    """
    from unittest.mock import MagicMock, patch

    # Set up environment variables
    monkeypatch.setenv("BFF_JAVA_VENDOR", "temurin")
    monkeypatch.setenv("BFF_JAVA_VERSION", "17")
    monkeypatch.setenv("BFF_JAVA_FETCH", "prefer")

    # Mock scyjava.config before importing _java_stuff
    mock_config = MagicMock()
    with patch("scyjava.config", mock_config):
        # Force reimport to trigger the env var logic
        if "bffile._java_stuff" in sys.modules:
            del sys.modules["bffile._java_stuff"]

        import bffile._java_stuff  # noqa: F401

        # Verify scyjava.config.set_java_constraints was called correctly
        mock_config.set_java_constraints.assert_called_once_with(
            vendor="temurin", version="17", fetch="prefer"
        )


@pytest.mark.skipif(
    "not config.getoption('--test-java-constraints')",
    reason="Slow integration test, use --test-java-constraints to run",
)
@pytest.mark.parametrize(
    "vendor,version",
    [("", "17"), ("", "21"), ("adoptium", "17"), ("temurin", "21"), ("zulu-jre", "11")],
)
def test_bff_java_constraints_integration(vendor: str, version: str) -> None:
    """Integration test: verify Java constraints work end-to-end.

    This is a slow integration test that downloads JDKs and JARs.
    Only run with --test-java-constraints flag.
    """
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

    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes for Java download
        check=False,  # Don't raise on error, handle manually
    )

    # If subprocess failed, print full stderr for debugging
    if result.returncode != 0:
        pytest.fail(
            f"Subprocess failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # Parse output
    output_lines = result.stdout.strip().split("\n")
    info = {}
    for line in output_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            info[key] = value

    java_version = info.get("VERSION", "")
    java_home = info.get("JAVA_HOME", "")
    java_vendor = info.get("VENDOR", "")

    assert java_version.startswith(f"{version}."), (
        f"Expected Java {version}, got {java_version}\n"
        f"JAVA_HOME: {java_home}\n"
        f"Vendor: {java_vendor}\n"
        f"Subprocess stderr: {result.stderr}"
    )
