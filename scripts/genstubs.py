import os
import shutil
import subprocess
from importlib import import_module

import cjdk
import jgo
import jpype.imports
import scyjava
import scyjava.config
from stubgenj import generateJavaStubs

with cjdk.java_env(vendor="temurin", version="11"):
    maven_url = "tgz+https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz"
    maven_dir = cjdk.cache_package("Maven", maven_url)
    maven_bin = next(maven_dir.rglob("mvn"))
    os.environ["PATH"] = os.pathsep.join([str(maven_bin.parent), os.environ["PATH"]])

    # doing this instead of scyjava.start_jvm because of
    # https://github.com/scijava/scyjava/issues/79
    endpoints = ["ome:formats-gpl", "ch.qos.logback:logback-classic:1.3.15"]
    _, workspace = jgo.resolve_dependencies(
        "+".join(endpoints),
        m2_repo=scyjava.config.get_m2_repo(),
        cache_dir=scyjava.config.get_cache_dir(),
        manage_dependencies=scyjava.config.get_manage_deps(),
        repositories=scyjava.config.get_repositories(),
    )
    jpype.addClassPath(os.path.join(workspace, "*"))
    jpype.startJVM(convertStrings=True)

    prefixes = ["loci.formats", "loci.common", "java", "ome.units", "ome.xml"]
    output = "stubs"
    generateJavaStubs(
        [import_module(prefix) for prefix in prefixes],  # type: ignore
        useStubsSuffix=True,
        outputDir=output,
        jpypeJPackageStubs=False,
        includeJavadoc=True,
    )

if shutil.which("ruff"):
    subprocess.run(
        [
            "ruff",
            "check",
            output,
            "--fix",
            "--unsafe-fixes",
            "--select=E,W,F,I,UP,C4,B,A001,RUF,TC,TID",
            "--ignore=E501,A001",
        ]
    )
    subprocess.run(["ruff", "format", "--target-version", "py39", output])
