# bffile

[![License](https://img.shields.io/pypi/l/bffile.svg?color=green)](https://github.com/tlambert03/bffile/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bffile.svg?color=green)](https://pypi.org/project/bffile)
[![Python
Version](https://img.shields.io/pypi/pyversions/bffile.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/bffile/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/bffile/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/bffile/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/bffile)

Yet another [Bio-Formats](https://github.com/ome/bioformats) wrapper for python

Bio-Formats is a Java library for reading and writing data in life sciences
image file formats. This library provides a python interface to Bio-Formats.

## Installation

```bash
pip install git+https://github.com/tlambert03/bffile
```

## Usage

```python
from bffile import BioFile

with BioFile("tests/data/ND2_dims_p4z5t3c2y32x32.nd2") as bf:
    print(bf.ome_metadata)  # ome_types.OME object
    print(bf.shape)  # shows full shape
    data = bf.to_numpy(series=1)
    print(data.shape, data.dtype)
```

## Notes on Java, Maven, and Bio-Formats

Because this library wraps the Java-based Bio-Formats library, it does require a
JVM at runtime.

> [!TIP]
> Every attempt is made to make that as easy (and customizable) as possible.
> **no additional action should be required by the user to use this package**.
>
> `pip install bffile` should be sufficient.

Here's how it works:

1. If you have Java installed, it will be detected and used.  This respects
   standard environment variables like `JAVA_HOME` and `PATH`.
2. If you don't have Java installed, the library will download a
   [JRE](https://en.wikipedia.org/wiki/Java_Runtime_Environment) using the
   [cjdk](https://pypi.org/project/cjdk/) package, and use it.
   - By default, it will use [zulu-jre](https://www.azul.com/downloads/) version
     11, but this can be customized by using the `JAVA_VENDOR` environment
     variable to specify a different vendor, and/or the `JAVA_VERSION`
     environment variable to specify a different version:

        ```sh
        export JAVA_VENDOR=temurin
        export JAVA_VERSION=17
        ```

3. [`jgo`](https://github.com/scijava/jgo) is used to fetch java dependencies at
   runtime.  This requires that `maven` is installed and on the `PATH`.  If you
   don't already have `maven` installed, `cjdk` will be used to download it as
   well.
   - If you would like to customize the version of `maven` that is fetched, you
   may use the `MAVEN_URL` and `MAVEN_SHA` variables.  for example:

        ```sh
        MAVEN_URL = "https://dlcdn.apache.org/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz"
        MAVEN_SHA = "a555254d6b53d267965a3404ecb14e53c3827c09c3b94b5678835887ab404556bfaf78dcfe03ba76fa2508649dca8531c74bca4d5846513522404d48e8c4ac8b"
        ```

4. By default, the latest version of Bio-Formats (GPL) is fetched from maven
   [maven](https://mvnrepository.com/artifact/ome/formats-gpl).  This
   can also be customized by using the `BIOFORMATS_VERSION` environment variable
   to specify a different version or license.  For example

    ```sh
    export BIOFORMATS_VERSION="6.13.0"
    # or
    export BIOFORMATS_VERSION="ome:formats-bsd:8.1"
    ```
