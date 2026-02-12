# bffile

[![License](https://img.shields.io/pypi/l/bffile.svg?color=green)](https://github.com/imaging-formats/bffile/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bffile.svg?color=green)](https://pypi.org/project/bffile)
[![Python Version](https://img.shields.io/pypi/pyversions/bffile.svg?color=green)](https://python.org)
[![CI](https://github.com/imaging-formats/bffile/actions/workflows/ci.yml/badge.svg)](https://github.com/imaging-formats/bffile/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imaging-formats/bffile/branch/main/graph/badge.svg)](https://codecov.io/gh/imaging-formats/bffile)

Modern Bio-Formats wrapper for python

### Documentation 📖

<https://imaging-formats.github.io/bffile/>

## Installation

```bash
pip install git+https://github.com/imaging-formats/bffile
```

## Usage

### Quick Start

```python
from bffile import BioFile
import numpy as np

with BioFile("tests/data/ND2_dims_p4z5t3c2y32x32.nd2") as bf:
    # Access OME metadata
    print(bf.ome_metadata)  # ome_types.OME object

    # Get lazy array for series 0 (no data loaded yet)
    arr = bf.as_array(series=0)
    print(arr.shape, arr.dtype)  # (T, C, Z, Y, X) shape

    # Index specific planes on-demand
    plane = arr[0, 0, 2]  # Only reads this plane
    roi = arr[:, :, :, 100:200, 50:150]  # Sub-regions

    # Materialize all data when needed
    full_data = np.array(arr)

    # Or use dask for lazy computation
    darr = bf.to_dask(series=0, chunks="auto")
    result = darr.mean(axis=2).compute()
```

For simple cases, use `imread()` to load a single series/resolution into memory:

```python
from bffile import imread

# Read series0/resolution0 into numpy array
# series and resolution parameters are optional and default to 0
data = imread("image.nd2", series=0, resolution=0)  
print(data.shape, data.dtype)  # (T, C, Z, Y, X) array
```

### Selecting Bio-Formats Version

Bio-Formats is downloaded at runtime (via [jgo](https://pypi.org/project/jgo/)).

By default, it will download the latest `ome:formats-gpl:RELEASE` maven artifact.
You can specify a different version by setting the `BIOFORMATS_VERSION` environment variable:

This variable accepts either a simple version string (e.g. `6.0.1`) or a full maven coordinate
(e.g. `ome:formats-gpl:6.0.1` or `ome:formats-bsd:7.3.1`):

```bash
# Use Bio-Formats 6.0.1 (GPL-licensed)
export BIOFORMATS_VERSION="6.0.1"

# Use BSD-licensed Bio-Formats 7.3.1
export BIOFORMATS_VERSION="ome:formats-bsd:7.3.1"
```

To see the currently installed version of Bio-Formats, you can check the
`BioFile.bioformats_version` static method:

```python
from bffile import BioFile

print(BioFile.bioformats_version())  # e.g. "8.1.1"
```

and to see the full maven coordinate that was used:

```python
from bffile import BioFile

print(BioFile.bioformats_maven_coordinate())  # e.g. "ome:formats-gpl:8.1.1"
```

> [!NOTE]
> We test back to version 6.0.1, but older versions may also work.  If you specifically need
> this code to work with an older version of bioformats, please open an issue.

### Java Runtime

> [!TIP]
> **No manual Java installation required!**  
>
> This package automatically downloads and manages the Java runtime using
> [cjdk](https://github.com/cachedjdk/cjdk) (via [scyjava](https://github.com/scijava/scyjava)).

By default, scyjava uses **Zulu JRE 11** (defined
[here](https://github.com/scijava/scyjava/blob/a2b8bed0a07a87d4c9b715a6dddaad308080f440/src/scyjava/config.py#L15-L20)).
You can configure the Java version and/or vendor using environment variables:

```bash
# Use Adoptium JDK 17
export BFF_JAVA_VERSION=17
export BFF_JAVA_VENDOR=adoptium

# Use Temurin JDK 21
export BFF_JAVA_VERSION=21
export BFF_JAVA_VENDOR=temurin
```

Available vendors: `zulu-jre`, `zulu`, `adoptium`, `temurin`, and
[other vendors listed in cjdk](https://github.com/cachedjdk/cjdk)

#### Java 8 Support

Bffile is *not* currently expected to work with Java 8, as `jpype` has
deprecated support for Java 8 as of `jpype` version 1.6. If you do want to try
with Java 8, you will minimally need to explicitly pin `jpype<1.6`.  If this is
an important use case for you, please open an issue to discuss Java 8 support.

## License

Licensing is a bit complicated for this project, so please read carefully.

- All code in this `bffile` repository is licensed under the [BSD-3-Clause
  License](./LICENSE/LICENSE).  You may use and distribute this code under the
  terms of the BSD-3-Clause License.
- However, a user who installs `bffile` from PyPI will, by default, end up with
  Java jars that are licensed under the GPL-2.0 License (read below). As such,
  **this package is listed on PyPI as GPL-2.0-or-later**.

When you run bffile for the first time, it will automatically download a number
of Java jars (via [`jgo`](https://pypi.org/project/jgo/)), each of which has its
own license.  By default, bffile downloads the
[`ome:formats-gpl:RELEASE`](https://mvnrepository.com/artifact/ome/formats-gpl)
maven artifact.

- `ome:formats-gpl` is licensed under the [GPLv2+
  License](./LICENSE/LICENSE_FORMATS_GPL)

If you would like to use bffile without any GPL-licensed jars, you can instead
opt into using
[`ome:formats-bsd`](https://mvnrepository.com/artifact/ome/formats-bsd) by
setting the `BIOFORMATS_VERSION` environment variable:

```
BIOFORMATS_VERSION="ome:formats-bsd"
```

- `ome:formats-bsd` is licensed under the [BSD-2-Clause
  License](./LICENSE/LICENSE_FORMATS_BSD)

If you need a package that defaults to BSD-licensed jars (and ships as
BSD-3 on PyPI), open an issue to discuss a `bffile-bsd` variant.
