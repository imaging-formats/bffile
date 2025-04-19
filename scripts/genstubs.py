#!/usr/bin/env python3
from __future__ import annotations
import os
import glob
import subprocess
import jpype
import jpype.imports
import stubgenj
from typing import List


def build_classpath(api_dir: str) -> List[str]:
    """
    Returns a list of all the JARs we need on the classpath:
      - the main formats-api JAR
      - all the copied-dependencies
    """
    target = os.path.join(api_dir, "target")
    # main JAR
    jars = glob.glob(os.path.join(target, "formats-api-*.jar"))
    # dependencies
    jars += glob.glob(os.path.join(target, "dependency", "*.jar"))
    return jars


def main() -> None:
    repo_root = os.path.abspath("bioformats/components/formats-api")
    if not os.path.isdir(repo_root):
        raise RuntimeError(
            "Please run this script from the project root, after cloning."
        )

    cp = build_classpath(repo_root)
    # start the JVM with our classpath and string conversion enabled
    jpype.startJVM(classpath=cp, convertStrings=True)
    jpype.imports.registerDomain("loci.formats")  # optional, for nicer imports

    # now import the classes you care about
    from loci.formats import ImageReader
    from loci.formats import (
        IFormatReader,
        IFormatHandler,
        IPyramidHandler,
        ICompressedTileReader,
    )

    # generate stubs for all of them (stubgenj also pulls in supertypes/interfaces)
    stubgenj.generateJavaStubs(
        [
            ImageReader,
            IFormatReader,
            IFormatHandler,
            IPyramidHandler,
            ICompressedTileReader,
        ],
        useStubsSuffix=True,  # makes top‑level folder look like “loci.formats-stubs/…”
        convertStrings=True,  # map java.lang.String → Python str
        outputDir="stubs",  # writes “stubs/loci/formats/…/*.pyi”
    )

    jpype.shutdownJVM()
    print("Done!  Stubs are in ./stubs/loci/formats/")


if __name__ == "__main__":
    main()
