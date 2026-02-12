---
icon: lucide/rocket
title: Get started
---

# Getting started with `bffile`

`bffile` is a **modern Bio-Formats wrapper for Python**.

It exposes the full power of Bio-Formats in a clean Pythonic API
backed by lazy data access.

**No special environment setup is required**, thanks to
[`scyjava`](https://github.com/scijava/scyjava),
[`jgo`](https://github.com/apposed/jgo),
[`jpype`](https://github.com/jpype-project/jpype), and
[`cjdk`](https://github.com/cachedjdk/cjdk):  
just `pip install bffile` and you're ready to go.

## Installation

```bash
pip install bffile
```

## Quick start

```python
import bffile

# load directly into memory
data = bffile.imread("path/to/file", series=0)

# or use lazy access to load only what you need
with bffile.BioFile("path/to/file") as biofile:
    # lazy series accessor
    lazy_array = biofile[0].as_array()
    # Load data into memory (T=0, C=1:4, Z=all, Y=100:200)
    data = lazy_array[0, 1:4, :, 100:200]
```

See [usage](usage.md) for more details and examples.
