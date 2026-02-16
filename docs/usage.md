---
title: Usage
icon: lucide/book-open
---

# Using `bffile`

## Quick Start

The simplest way to read an image is with [`imread`][bffile.imread]:

```python
from bffile import imread

data = imread("image.nd2")
print(data.shape, data.dtype)
# (10, 2, 5, 512, 512) uint16

# or, to read a specific series/resolution-level:
data = imread("image.nd2", series=1, resolution=0)
```

This reads the specified series/resolution into memory as a numpy array with
shape `(T, C, Z, Y, X)`. For most other use cases, you'll want more control —
that's where [`BioFile`][bffile.BioFile] comes in.

```python
from bffile import BioFile

with BioFile("image.nd2") as bf:
    arr = bf.as_array()       # lazy array accessor
    plane = arr[0, 0, 2]      # indexing is just a view — no data read yet!
    data = np.asarray(plane)  # call np.asarray to read the data into memory
```

For info on extracting data, see:

- [The `LazyBioArray` API](#reading-data-with-lazybioarray) for reading pixel
  data with lazy loading and sub-region slicing.
- [Using as a zarr store](#complete-virtual-ome-zarr-view) for interoperability
  with the OME-Zarr ecosystem without file conversion.
- [Using dask for lazy computation](#lazy-computation-with-dask) for wrapping
  the lazy array in a dask array for parallelized computations.
- [Presenting as an xarray
  DataArray](#labeled-dimensionscoordinates-with-xarray) which makes it easy to
  work with labeled dimensions and coordinates parsed from the OME metadata.

## Opening Files with BioFile

[`BioFile`][bffile.BioFile] manages the lifecycle of the underlying Java reader and the
associated file handle. The recommended pattern is a context manager:

```python
with BioFile("image.nd2") as bf:
    data = bf.read_plane()
# reader fully cleaned up here
```

You can also manage the lifecycle explicitly:

```python
bf = BioFile("image.nd2")
bf.open()
# ... use bf ...
bf.close()    # release file handle (fast reopen later)
bf.open()     # cheap — just reacquires the file handle
# ... use bf again ...
bf.destroy()  # full cleanup (or let GC handle it)
```

!!! danger "Critical: Some operations require an open file"

    `BioFile` does *not* attempt to magically open/close files for you as needed.
    However, some methods like [`as_array()`](#reading-data-with-lazybioarray) and
    [`to_dask()`](#using-dask-for-lazy-computation) return objects that require the
    file to be open when indexed or computed. You are responsible for ensuring the
    file is open while using those objects.

    ```python
    from bffile import BioFile

    with BioFile("image.nd2") as bf:
        arr = bf.as_array()

    try:
        arr[0, 0, 2]  # ERROR!!
    except RuntimeError as e:
        print(e)  # "File not open - call open() first"
    ```

### Understanding Lifecycle

`BioFile` has three states:

1. `UNINITIALIZED`: The Python `BioFile` object exists, but the file handle is not open, and no Java resources are allocated.
2. `OPEN`: The file is open and the Java reader is initialized. You can read data and metadata.
3. `SUSPENDED`: The file handle is released but the Java reader and all parsed metadata remain in memory. You cannot read data, but you can still access metadata.  Re-opening the file from this state is fast.

``` mermaid
---
title: BioFile Lifecycle
---
stateDiagram-v2
    direction LR
    [*] --> UNINITIALIZED : <code>\_\_init__()</code>
    UNINITIALIZED --> OPEN : <code>open()</code>
    OPEN --> SUSPENDED : <code>close()</code>
    SUSPENDED --> OPEN : <code>open()</code>
    OPEN --> UNINITIALIZED : <code>destroy()</code>
    SUSPENDED --> UNINITIALIZED : <code>destroy()</code>
```

| Transition | What happens |
| --- | --- |
| [`__init__()`][bffile.BioFile] | Creates the `BioFile` object but does not open the file or initialize the reader. |
| [`open()`][bffile.BioFile.open] (first call) | Full initialization — format detection, header parsing (`setId` in Java). Slow. |
| [`close()`][bffile.BioFile.close] | Releases the OS file handle but keeps all parsed state in memory. |
| [`open()`][bffile.BioFile.open] (after `close()`) | Just reopens the file handle (`reopenFile` in Java). Fast. |
| [`destroy()`][bffile.BioFile.destroy] / [`__exit__()`][bffile.BioFile.__exit__] | Full teardown — Java reader and all cached state released. |

`close()` is lightweight: metadata (via `core_metadata()`, `len()`,
etc.) remains accessible while the file handle is released. This is
useful when you want to avoid holding file descriptors open but plan to
read more data later.

!!! tip "Context manager vs explicit close"
    The context manager (`with`) calls `destroy()` on exit — full cleanup.
    If you want the fast-reopen behavior, use explicit `open()` / `close()`
    calls instead.

!!! question "Memoization"

    **Memoization speeds up *future* calls to `open()`, from an uninitialized
    state, even across different Python sessions.**

    The [`memoize`][bffile.BioFile] parameter controls whether the
    initialized reader state is cached to a `.bfmemo` file *on disk*. This can
    improve performance for the `UNINITIALIZED → OPEN` transition (i.e., when the
    java reader is fully initialized from scratch) for *subsequent* reads of the
    same file in a new Python session, or after `destroy()` has been called.

    ```python
    # First open: full init + saves .bfmemo file to disk
    with BioFile("image.nd2", memoize=True) as bf:
        ...

    # Subsequent opens are faster: loads from .bfmemo instead of re-parsing
    with BioFile("image.nd2", memoize=True) as bf:
        ...
    ```

    You *must* have `memoize=True` on both the initial open and subsequent
    opens for this to work.

    !!! info "BIOFORMATS_MEMO_DIR"
        By default, the `.bfmemo` file is saved in the same directory as the
        original image. You must have write permission to this directory.
        You can change this with the `BIOFORMATS_MEMO_DIR` [environment
        variable](#environment-variables).

## The Series Data Model

Bio-Formats models files as a sequence of __series__ (e.g., wells in a plate,
fields of view, tiles in a mosaic, etc...). Each series is a 5D dataset with shape
`(T, C, Z, Y, X)`, and may have multiple __resolution__ levels (pyramid
layers).

!!! info "Mental model"
    ```sh
    BioFile                    # Container (usually a file, but possibly multiple)
    ├── Series 0               # e.g., first field of view
    │   ├── Resolution 0       # full-resolution data
    │   └── Resolution 1       # downsampled pyramid level (if present)
    ├── Series 1
    │   └── ...
    └── ...
    ```

### `bffile` is Stateless

If you're familiar with the Bio-Formats Java API, you will be used
to using `setSeries` to change the active series before following
up with calls to read data or metadata.

`bffile.BioFile` aims for a __stateless__ API: all methods that pertain to
a specific series or resolution level take an explicit
`series` argument and an optional `resolution` level.  Omitting these
arguments defaults to `series=0` and `resolution=0`.  As a convenience,
`BioFile` also provides a [`Series`][bffile.Series] proxy object, described below.

### Accessing Series

A [`Series`][bffile.Series] object is a lightweight proxy that pre-fills
the `series=` argument on all calls back to the parent
[`BioFile`][bffile.BioFile]:

[`BioFile`][bffile.BioFile] implements `Sequence[bffile.Series]`, so you can
index, and iterate:

```python
with BioFile("multi_scene.czi") as bf:
    print(len(bf))         # number of series

    s = bf[0]              # first series
    print(s.shape)             # (10, 2, 5, 512, 512)
    print(s.dtype)             # uint16
    print(s.is_rgb)            # False
    print(s.resolution_count)  # 1

    s = bf[-1]             # last series
    first_two = bf[0:2]    # slice returns list[Series]

    for series in bf:      # iterate over all series
        print(series.shape, series.dtype)
```

`Series` objects also have methods that mirror those on `BioFile`, but with
the `series` argument pre-filled:

```python
with BioFile("image.nd2") as bf:
    s = bf[0]
    arr = s.as_array()         # same as bf.as_array(series=0)
    meta = s.core_metadata()   # same as bf.core_metadata(series=0)
```

!!! danger "critical"
    The `BioFile` [must be open](#understanding-lifecycle) while you use
    any `Series` objects obtained from it.  If you don't want to use the context
    manager, you should manage the lifecycle explicitly with `open()` and
    `close()`.

## Reading Data with LazyBioArray

The recommended way to read pixel data is through
[`LazyBioArray`][bffile.LazyBioArray], obtained via
[`as_array()`][bffile.BioFile.as_array].  This object behaves like a numpy array
but reads the minimal amount of data from disk when you index into it (including
sub-plane/XY slicing).

```python
with BioFile("image.nd2") as bf:
    arr = bf[0].as_array()  # no data read yet!
    print(arr)
    # LazyBioArray(shape=(10, 2, 5, 512, 512), dtype=uint16, file='image.nd2')
```

No data is loaded when you create the array. Indexing creates lazy views
_without reading data_. Data is read from disk only when you materialize the
view with `np.asarray()` or numpy other operations:

```python
with BioFile("image.nd2") as bf:
    arr = bf[0].as_array()  # no reading yet!

    # create a lazy view of a single plane (t=0, c=0, z=2)
    plane_view = arr[0, 0, 2]              # LazyBioArray, shape: (512, 512)
    plane = np.asarray(plane_view)         # NOW data is read from disk

    # create lazy view of all timepoints for one channel and z-slice
    timeseries_view = arr[:, 0, 2]         # LazyBioArray, shape: (10, 512, 512)
    timeseries = np.asarray(timeseries_view)  # reads data

    # lazy view of a (100, 100) sub-region within the YX plane
    # in the third timepoint, for all channels, and the first z-slice
    roi_view = arr[2, :, 0, 100:200, 50:150]  # LazyBioArray, shape: (2, 100, 100)
    roi = np.asarray(roi_view)             # only the requested pixels are read

    # materialize the full dataset
    full = np.asarray(arr)                 # shape: (10, 2, 5, 512, 512)
```

LazyBioArray indexing creates lazy views with the following behavior:

- __Integer indexing__ squeezes that dimension: `arr[0, 0, 2]` returns a
  LazyBioArray view with shape `(Y, X)` instead of `(1, 1, 1, Y, X)`.
- __Slice indexing__ keeps the dimension: `arr[0:1, 0:1, 2:3]` returns a view
  with shape `(1, 1, 1, Y, X)`.
- __Ellipsis__: `arr[..., 100:200, 50:150]` works as expected.
- __Negative indices__: `arr[-1]` creates a view of the last timepoint.

!!! warning "Unsupported indexing"
    Step slicing (`arr[::2]`), fancy indexing (`arr[[0, 2]]`), and
    boolean masks (`arr[arr > 100]`) are __not__ supported and will raise
    `NotImplementedError`.

### Sub-region reads are efficient

When you slice the Y or X dimensions and then materialize the view, only the
requested pixels are read from disk — not the full plane:

```python
# create a view of a 100x100 pixel region
roi_view = arr[:, :, :, 200:300, 300:400]  # lazy view, no I/O yet
roi = np.asarray(roi_view)  # reads only the 100x100 region from each plane
```

This makes `LazyBioArray` well-suited for exploring large images without
loading everything into memory.

### Numpy integration

`LazyBioArray` implements the `__array__` protocol, so you can pass it
directly to numpy functions:

```python
full_data = np.array(arr)           # materialize to ndarray
max_proj = np.max(arr, axis=2)      # z-projection (reads all data)
```

!!! tip "Keep the file open"
    The parent `BioFile` must remain open while you use `LazyBioArray`.
    Always use it inside the `with` block (or between explicit
    `open()`/`close()` calls).

## Third-party array types

We support casting `LazyBioArray` to various third-party array types
for interoperability with their ecosystems:

- [`to_xarray()`][bffile.BioFile.to_xarray] → [`xarray.DataArray`](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataarray)
- [`to_zarr_store()`][bffile.BioFile.to_zarr_store] → [`zarr.abc.store.Store`](https://zarr.readthedocs.io/en/v3.1.2/user-guide/storage.html)
- [`to_dask()`](#using-dask-for-lazy-computation) → [`dask.array.Array`](https://docs.dask.org/en/stable/array.html)

You will find each of these methods on

- `BioFile`: where you can specify `series` and `resolution` as arguments
- `Series`: where the `series` argument is pre-filled, and you can specify `resolution` if needed.
- `LazyBioArray`: where the returned object shares the same lazy view onto the data.

### Labeled dimensions/coordinates with `xarray`

[`xarray`](https://docs.xarray.dev/en/stable/) provides a powerful data
structure for working with labeled, multi-dimensional arrays.  Bioforamts data
is naturally represented as an `xarray.DataArray` with dimensions (`"T", "C",
"Z", "Y", "X"`, and sometimes `"S"` for RGB).  Metadata is parsed and used to
populate the `.dims` and `.coords` attributes.  As you index into the array, the
lazy reading behavior is preserved, so you can explore large datasets without
loading everything into memory.  Semantics for coords are as follows:

- `T`: `delta_t` coordinate taken from OME `pixels.planes`
- `C`: Channel names from OME `pixels.channels` (e.g. "DAPI", "FITC", etc...)
- `Z`: `delta_z` coordinate taken from OME `pixels.planes`
- `Z`, `Y`, `X`: physical pixel sizes from OME `pixels.physical_size_...`
- `S`: RGB/RGBA channels, if applicable.

```python
with BioFile("image.nd2") as bf:
    xarr = bf.to_xarray(series=0)  # xarray.DataArray with dims and coords
    print(xarr.dims)       # ('T', 'C', 'Z', 'Y', 'X')
    print(xarr.coords)     # coordinates parsed from OME metadata

    # indexing still creates lazy views
    plane_view = xarr[0, 0, 2]  # xarray.DataArray view of a single plane
    assert "T" not in plane_view.dims  # dimension is squeezed out by integer indexing

    # you can also use named indexing with .sel and .isel
    plane_view = xarr.isel(T=0, C=0, Z=2)  # same plane view using named indexing
    red_channel = xarr.sel(C="Widefield Red")  # select by channel name (if available)

    # the full ome-types.OME metadata is available in the .attrs of the DataArray
    ome = xarr.attrs["ome_metadata"]
```

### Complete virtual OME-Zarr view

The [`to_zarr_store()`][bffile.BioFile.to_zarr_store] method returns a
`zarr.Store` that can be passed to `zarr.open()`.   When you cast a
complete `BioFile` to a zarr store, without specifying a series or resolution,
__the returned store provides a virtual view of the entire file as a spec compliant
OME-Zarr__, (similar to what you would get if you converted the file to
using [`bioformats2raw`](https://github.com/glencoesoftware/bioformats2raw), but
without requiring any conversion or additional disk space).

Groups follow the [`bioformats2raw.layout` transitional
spec](https://ngff.openmicroscopy.org/specifications/0.5/index.html#bioformats2raw-layout-transitional)

!!! warning "It's not 'free'"
    While viewing any bioformats-supported file as an OME-Zarr without conversion is
    a powerful feature: you should _not_ assume that you will get the anywhere near
    same performance as a native OME-Zarr directory store. Performance will depend
    entirely on the native file structure, and many will not be as optimized for
    chunked access as a purpose-built, rechunked and compressed OME-Zarr.

    _This pattern is intended for quick interoperability and convenience,
    not for high performance._

```python
from bffile import open_ome_zarr_group
import zarr

ome_zarr = open_ome_zarr_group("image.nd2")
assert isinstance(ome_zarr, zarr.Group)

ome_meta = ome_zarr.attrs["ome"]
assert "bioformats2raw.layout" in ome_meta

series0 = ome_zarr["0"]  # group for series 0
assert "multiscales" in series0.attrs["ome"]  # multiscales metadata is present
level0 = series0["0"]    # array for resolution level 0
assert isinstance(level0, zarr.Array)
print(level0.shape, level0.dtype)
```

### Lazy computation with `dask`

For computations over large datasets, [`BioFile.to_dask`][bffile.BioFile.to_dask]
wraps `LazyBioArray` in a dask array:

```python
with BioFile("image.nd2") as bf:
    darr = bf.to_dask(chunks=(1, 1, 1, -1, -1))
    result = darr.mean(axis=2).compute()  # lazy z-projection
```

You don't gain any _additional_ data reading "laziness" here.  But you can use
dask's rich ecosystem of chunked, parallelized computations and out-of-core
algorithms on top of the lazy reading provided by `LazyBioArray`.

!!! danger "File must be open"
    Remember that the `BioFile` [must be open](#understanding-lifecycle)
    when you `.compute()` the dask array.

You can also use tile-based chunking for very large planes:

```python
with BioFile("image.nd2") as bf:
    darr = bf.to_dask(tile_size=(512, 512))       # explicit tile size
    darr = bf.to_dask(tile_size="auto")           # query Bio-Formats for optimal size
```

!!! note "Dask is optional"
    Dask is not installed by default. Install it with:

    ```sh
    pip install dask

    # or, to get a version that we guarantee is compatible with bffile
    # install the extra:
    pip install bffile[dask]
    ```

## Metadata

### OME Metadata

[`ome_metadata()`][bffile.BioFile.ome_metadata] returns a rich, structured
[`ome_types.OME`][] object, with all of the metadata parsed and organized
according to the OME data model.

```python
with BioFile("image.nd2") as bf:
    ome = bf.ome_metadata        # parsed OME object
    xml_str = bf.ome_xml         # raw OME-XML string

    # explore structured metadata
    print(ome.images[0].pixels.size_x)
    print(ome.images[0].pixels.physical_size_x)
```

### Core Metadata

[`core_metadata()`][bffile.BioFile.core_metadata] returns a
[`CoreMetadata`][bffile.CoreMetadata] dataclass with `shape`, `dtype`, and
acquisition flags for a given series/resolution:

```python
with BioFile("image.nd2") as bf:
    meta = bf.core_metadata(series=0, resolution=0)
    print(meta.shape)             # OMEShape(t=10, c=2, z=5, y=512, x=512, rgb=1)
    print(meta.dtype)             # uint16
    print(meta.dimension_order)   # "XYCZT"
    print(meta.is_little_endian)  # True
```

### Global Metadata

[`global_metadata()`][bffile.BioFile.global_metadata] returns
reader/file-specific key/value pairs:

```python
with BioFile("image.nd2") as bf:
    for key, value in bf.global_metadata().items():
        print(f"{key}: {value}")
```

## Reader Discovery

You can query Bio-Formats for supported formats without opening a file:

```python
from bffile import BioFile

# Bio-Formats version
print(BioFile.bioformats_version())  # "8.1.1"

# all supported file extensions
suffixes = BioFile.list_supported_suffixes()  # {"nd2", "czi", "tiff", ...}

# detailed reader info
for reader in BioFile.list_available_readers():
    print(f"{reader.format}: {reader.suffixes} (GPL={reader.is_gpl})")
```

## Environment Variables

| <div style="width: 130px;">Variable</div> | Description | <div style="width: 170px;">Default</div> |
| -------- | ----------- | ------- |
| `BIOFORMATS_VERSION` | Bio-Formats version or full Maven coordinate (e.g. `"7.0.0"` or `"ome:formats-gpl:7.0.0"`) | `"ome:formats-gpl:RELEASE"` |
| `BIOFORMATS_MEMO_DIR` | Directory for `.bfmemo` cache files | same as file |
| `BFF_JAVA_VERSION` | Java version to use (e.g. `11`, `17`, `21`) | `11` |
| `BFF_JAVA_VENDOR` | Java vendor (e.g. `zulu-jre`, `temurin`, `adoptium`) | `zulu-jre` |
| `BFF_JAVA_FETCH` | Java fetch behavior: `always`, `never`, or `auto` (See [scyjava docs](https://github.com/scijava/scyjava?tab=readme-ov-file#bootstrap-a-java-installation)). | `always` |
