#!/usr/bin/env -S uv run
# /// script
# requires-python = "==3.13"
# dependencies = [
#     "cjdk",
#     "scyjava",
#     "stubgenj",
#     "ruff",
# ]
# ///
"""Minimal scyjava stub generator script.

Usage:
    uv run stubgen.py org.apache.commons:commons-lang3:3.12.0
    uv run stubgen.py org.apache.commons:commons-lang3:3.12.0 --prefix org.apache
    uv run stubgen.py --output-dir ./my-stubs org.myproject:myproject:1.0.0
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from importlib import import_module
from itertools import chain
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from zipfile import ZipFile

import cjdk
import scyjava
import scyjava.config
from stubgenj import generateJavaStubs  # pyright: ignore

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("scyjava_stubs")


def generate_stubs(
    endpoints: Sequence[str],
    prefixes: Sequence[str] = (),
    output_dir: str | Path = "./typings",
    convert_strings: bool = False,
    include_javadoc: bool = True,
) -> None:
    """Generate stubs for the given maven endpoints.

    Parameters
    ----------
    endpoints : Sequence[str]
        The maven endpoints to generate stubs for. This should be a list of GAV
        coordinates, e.g. ["org.apache.commons:commons-lang3:3.12.0"].
    prefixes : Sequence[str], optional
        The prefixes to generate stubs for. This should be a list of Java class
        prefixes that you expect to find in the endpoints. For example,
        ["org.apache.commons"].  If not provided, the prefixes will be
        automatically determined from the jar files provided by endpoints.
    output_dir : str | Path, optional
        The directory to write the generated stubs to. Defaults to "./typings".
    convert_strings : bool, optional
        Whether to cast Java strings to Python strings in the stubs. Defaults to True.
    include_javadoc : bool, optional
        Whether to include Javadoc in the generated stubs. Defaults to True.
    """
    import jpype.imports

    startJVM = jpype.startJVM

    scyjava.config.endpoints.extend(endpoints)

    # Use scyjava's default Java setup via cjdk
    with cjdk.java_env():

        def _patched_start(*args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("convertStrings", convert_strings)
            startJVM(*args, **kwargs)

        with patch.object(jpype, "startJVM", new=_patched_start):
            scyjava.start_jvm()

        _prefixes = set(prefixes)
        if not _prefixes:
            cp = jpype.getClassPath(env=False)
            ep_artifacts = tuple(ep.split(":")[1] for ep in endpoints)
            for j in cp.split(os.pathsep):
                if Path(j).name.startswith(ep_artifacts):
                    _prefixes.update(list_top_level_packages(j))

        prefixes = sorted(_prefixes)
        logger.info(f"Using endpoints: {scyjava.config.endpoints!r}")
        logger.info(f"Generating stubs for: {prefixes}")
        logger.info(f"Writing stubs to: {output_dir}")

        jmodules = [import_module(prefix) for prefix in prefixes]
        generateJavaStubs(
            jmodules,
            useStubsSuffix=False,
            outputDir=str(output_dir),
            jpypeJPackageStubs=False,
            includeJavadoc=include_javadoc,
        )

    output_dir = Path(output_dir)
    ruff_check(output_dir.absolute())
    ruff_format(output_dir.absolute())


def ruff_check(output: Path) -> None:
    """Run ruff check and format on generated stub files."""
    py_files = [str(x) for x in chain(output.rglob("*.py"), output.rglob("*.pyi"))]
    if shutil.which("ruff"):
        logger.info(
            "Running ruff check on %d generated stubs in %s",
            len(py_files),
            str(output),
        )
        subprocess.run(
            [
                "ruff",
                "check",
                *py_files,
                "--target-version",
                "py310",
                "--quiet",
                "--fix-only",
                "--unsafe-fixes",
                "--select=E,W,F,I,UP,RUF,TC,TID",
            ],
            check=False,
        )
        logger.info("Running ruff format")
        subprocess.run(["ruff", "format", *py_files, "--quiet"], check=False)


def ruff_format(output: Path) -> None:
    """Run ruff format on generated stub files."""
    py_files = [str(x) for x in chain(output.rglob("*.py"), output.rglob("*.pyi"))]
    if shutil.which("ruff"):
        logger.info(
            "Running ruff format on %d generated stubs in %s",
            len(py_files),
            str(output),
        )
        subprocess.run(["ruff", "format", *py_files, "--quiet"], check=False)


def list_top_level_packages(jar_path: str) -> set[str]:
    """Inspect a JAR file and return the set of top-level Java package names."""
    packages: set[str] = set()
    with ZipFile(jar_path, "r") as jar:
        # find all classes
        class_dirs = {
            entry.parent
            for x in jar.namelist()
            if (entry := PurePath(x)).suffix == ".class"
        }

        roots: set[PurePath] = set()
        for p in sorted(class_dirs, key=lambda p: len(p.parts)):
            # If none of the already accepted roots is a parent of p, keep p
            if not any(root in p.parents for root in roots):
                roots.add(p)
        packages.update({str(p).replace("/", ".") for p in roots})

    return packages


def main() -> None:
    """The main entry point for the stub generator script."""
    logging.basicConfig(level="INFO")
    parser = argparse.ArgumentParser(
        description="Generate Python Type Stubs for Java classes."
    )
    parser.add_argument(
        "endpoints",
        type=str,
        nargs="+",
        help="Maven endpoints to install and use (e.g. org.myproject:myproject:1.0.0)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="package prefixes to generate stubs for (e.g. org.myproject), "
        "may be used multiple times. If not specified, prefixes are gleaned from the "
        "downloaded artifacts.",
        action="append",
        default=[],
        metavar="PREFIX",
        dest="prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./typings",
        help="Filesystem path to write stubs to. Defaults to './typings'.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    generate_stubs(
        endpoints=args.endpoints,
        prefixes=args.prefix,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
