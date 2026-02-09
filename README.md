# bffile

[![License](https://img.shields.io/pypi/l/bffile.svg?color=green)](https://github.com/tlambert03/bffile/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bffile.svg?color=green)](https://pypi.org/project/bffile)
[![Python Version](https://img.shields.io/pypi/pyversions/bffile.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/bffile/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/bffile/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/bffile/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/bffile)

Yet another Bio-Formats wrapper for python

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

Bffile is not currently tested with Java 8, there are some known issues with
casting to numpy.  If you do want to try with Java 8, you will minimally need to
explicitly pin `jpype<1.6`.

## License

Licensing is a bit complicated for this project, so please read carefully.

- All code in this `bffile` repository is licensed under the [BSD-3-Clause
  License](./LICENSE/LICENSE).  You may use and distribute this code under the
  terms of the BSD-3-Clause License.
- However, a user who installs `bffile` from PyPI will, by default, end up with
  Java jars that are licensed under the GPL-2.0 License (read below). As such,
  this package is listed on PyPI as GPL-2.0-or-later.

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
