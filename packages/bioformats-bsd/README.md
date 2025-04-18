# bioformats

[![License](https://img.shields.io/pypi/l/bioformats.svg?color=green)](https://github.com/tlambert03/bioformats/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bioformats.svg?color=green)](https://pypi.org/project/bioformats)
[![Python Version](https://img.shields.io/pypi/pyversions/bioformats.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/bioformats/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/bioformats/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/bioformats/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/bioformats)

Yet another Bio-Formats wrapper for python

## Installation

```bash
pip install git+https://github.com/tlambert03/bioformats
```

### Install Java ...

This package requires that you have java installed.  
[INSERT GUIDELINES HERE]

## Usage

```python
from bioformats import BioFile

with BioFile("tests/data/ND2_dims_p4z5t3c2y32x32.nd2") as bf:
    print(bf.ome_metadata)  # ome_types.OME object
    print(bf.shape)  # shows full shape
    data = bf.to_numpy(series=1)
    print(data.shape, data.dtype)
```
