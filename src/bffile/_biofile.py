from __future__ import annotations

import os
import sys
import warnings
import weakref
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload

import jpype
import numpy as np
from ome_types import OME

from bffile._core_metadata import CoreMetadata
from bffile._series import Series

from . import _utils
from ._java_stuff import jtype_to_python
from ._jimports import jimport

if TYPE_CHECKING:
    from collections.abc import Iterator

    import dask.array
    import java.lang
    from loci.formats import IFormatReader
    from typing_extensions import Self

    from bffile._lazy_array import LazyBioArray


@dataclass(frozen=True)
class ReaderInfo:
    """Information about a Bio-Formats reader class.

    Attributes
    ----------
    format : str
        Human-readable format name (e.g., "Nikon ND2").
    suffixes : tuple[str, ...]
        Supported file extensions (e.g., ("nd2", "jp2")).
    class_name : str
        Full Java class name (e.g., "ND2Reader").
    is_gpl : bool
        Whether this reader requires GPL license (True) or is BSD (False).
    """

    format: str
    suffixes: tuple[str, ...]
    class_name: str
    is_gpl: bool


# by default, .bfmemo files will go into the same directory as the file.
# users can override this with BIOFORMATS_MEMO_DIR env var
BIOFORMATS_MEMO_DIR: Path | None = None
_BFDIR = os.getenv("BIOFORMATS_MEMO_DIR")
if _BFDIR:
    BIOFORMATS_MEMO_DIR = Path(_BFDIR).expanduser().absolute()
    BIOFORMATS_MEMO_DIR.mkdir(exist_ok=True, parents=True)

# Java byte array size limit: 2^31 - 8 (leaves room for array header)
# Bio-Formats will fail with "Array size too large" if we exceed this.
# Key insight: This is a HARD limit in Java - can't be increased without JVM changes.
# Solution: Automatic tiling when reading >2GB planes (transparent to users)
MAX_JAVA_ARRAY_SIZE: int = 2**31 - 8
if _max_bytes := os.getenv("BIOFORMATS_MAX_JAVA_BYTES"):  # pragma: no cover
    try:
        MAX_JAVA_ARRAY_SIZE = int(_max_bytes)
    except ValueError:
        warnings.warn(
            f"Invalid BIOFORMATS_MAX_JAVA_BYTES: {_max_bytes!r}. "
            f"Using default {MAX_JAVA_ARRAY_SIZE}",
            stacklevel=2,
        )


class BioFile(Sequence[Series]):
    """Read image and metadata from file supported by Bioformats.

    BioFile instances must be explicitly opened before use, either by:

    1. Using a context manager: `with BioFile(path) as bf: ...`
    2. Explicitly calling `open()`: `bf = BioFile(path).open()`

    The recommended pattern is to use the context manager, which automatically
    handles opening and closing the file and cleanup of Java resources; but many usage
    patterns *will* also require explicit open/close.

    BioFile instances are not thread-safe. Create separate instances per thread.

    Lifecycle
    ---------
    BioFile manages the underlying Java reader through three states:

        UNINITIALIZED ── open() ──> OPEN ── close() ──> SUSPENDED
             ↑    ↑                  │ ↑                     │
             │    └── destroy() ─────┘ └──── open() ─────────┘
             └─────── destroy() ─────────────────────────────┘

    - `open()` first call: full initialization via `Java:setId()` (slow).
    - `close()`: releases file handles but preserves all parsed metadata
      and reader state in memory (`Java:close(fileOnly=true)`).
    - `open()` after `close()`: fast: simply reacquires the file handle — no re-parsing.
    - `destroy()` / `__exit__()`: full teardown — releases the Java
      reader and all cached state, returning to `UNINITIALIZED`.  `open()`
      can be called again but will require full re-initialization.
    - `__del__` GC finalizer: equivalent to `destroy()`.

    `SUSPENDED` preserves the Java reader's parsed state (format-specific
    headers, CoreMetadata, OME-XML DOM, metadata hashtable) in JVM heap,
    as well as Python-side `core_metadata()`. Only file handles are released.
    This is what enables the fast `open()` path.

    `destroy()` releases all of these, making the Java objects eligible for
    JVM garbage collection. The `BioFile` reverts to its initial state.

    Parameters
    ----------
    path : str or Path
        Path to file
    meta : bool, optional
        Whether to get metadata as well, by default True
    original_meta : bool, optional
        Whether to also retrieve the proprietary metadata as structured annotations in
        the OME output, by default False
    memoize : bool or int, optional
        Threshold (in milliseconds) for memoizing the reader. If the time
        required to call `reader.setId()` is larger than this number, the initialized
        reader (including all reader wrappers) will be cached in a memo file, reducing
        time to load the file on future reads. By default, this results in a hidden
        `.bfmemo` file in the same directory as the file. The `BIOFORMATS_MEMO_DIR`
        environment can be used to change the memo file directory.
        Set `memoize` to greater than 0 to turn on memoization, by default it's off.
        https://downloads.openmicroscopy.org/bio-formats/latest/api/loci/formats/Memoizer.html
    options : dict[str, bool], optional
        A mapping of option-name -> bool specifying additional reader-specific options.
        See: https://docs.openmicroscopy.org/bio-formats/latest/formats/options.html
        For example: to turn off chunkmap table reading for ND2 files, use
        `options={"nativend2.chunkmap": False}`
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        meta: bool = True,
        original_meta: bool = False,
        memoize: int | bool = 0,
        options: dict[str, bool] | None = None,
    ):
        self._path = str(path)
        self._lock = RLock()
        self._meta = meta
        self._original_meta = original_meta
        self._memoize = memoize
        self._options = options

        # Reader and finalizer created in open()
        self._java_reader: IFormatReader | None = None
        # 2D structure: list[series][resolution]
        self._core_meta_list: list[list[CoreMetadata]] | None = None
        self._finalizer: weakref.finalize | None = None
        self._suspended: bool = False

    def _ensure_java_reader(self) -> IFormatReader:
        """Return the native reader, raising if not open."""
        if self._java_reader is None or self._suspended:
            raise RuntimeError("File not open - call open() first")
        return self._java_reader

    def _get_core_metadata(self, reader: IFormatReader) -> list[list[CoreMetadata]]:
        """Parse flat CoreMetadata list into 2D structure.

        Bio-Formats returns metadata as a flat list where entries are organized
        as: [series0_res0, series0_res1, ..., series1_res0, series1_res1, ...].
        The first entry of each series has resolution_count set to indicate how
        many resolution levels that series has.
        """
        # Cache metadata in 2D structure: list[series][resolution]
        # Bio-Formats returns a flat list where the first entry of each
        # series has resolutionCount set. We parse this into 2D.
        flat_list = [CoreMetadata.from_java(x) for x in reader.getCoreMetadataList()]

        result: list[list[CoreMetadata]] = []
        i = 0
        while i < len(flat_list):
            resolution_count = flat_list[i].resolution_count
            result.append(flat_list[i : i + resolution_count])
            i += resolution_count
        return result

    def core_metadata(self, series: int = 0, resolution: int = 0) -> CoreMetadata:
        """Get metadata for specified series and resolution.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0

        Returns
        -------
        CoreMetadata
            Metadata for the specified series and resolution

        Raises
        ------
        RuntimeError
            If file is not open
        IndexError
            If series or resolution index is out of bounds

        Notes
        -----
        Resolution support is included for future compatibility, but currently
        only resolution 0 (full resolution) is exposed in the public API.
        """
        if self._core_meta_list is None:
            raise RuntimeError("File not open - call open() first")
        if series < 0 or series >= len(self._core_meta_list):
            raise IndexError(
                f"Series index {series} out of range "
                f"(file has {len(self._core_meta_list)} series)"
            )
        if resolution < 0 or resolution >= len(self._core_meta_list[series]):
            raise IndexError(
                f"Resolution index {resolution} out of range "
                f"(series {series} has {len(self._core_meta_list[series])} "
                f"resolution levels)"
            )
        return self._core_meta_list[series][resolution]

    def global_metadata(self) -> Mapping[str, Any]:
        """Return the global metadata as a dictionary.

        This includes all metadata that is not specific to a particular series or
        resolution. The exact contents will depend on the file and reader, but may
        include things like instrument information, acquisition settings, and other
        annotations.

        Returns
        -------
        Mapping[str, Any]
            Global metadata key-value pairs

        Raises
        ------
        RuntimeError
            If file is not open
        """
        reader = self._ensure_java_reader()
        meta = reader.getGlobalMetadata()
        return {str(k): jtype_to_python(v) for k, v in meta.items()}

    def open(self) -> Self:
        """Open file and initialize reader, or re-open if previously closed.

        On first call, performs full initialization (`setId`). If the file
        was previously closed via `close()`, reopens cheaply by reacquiring
        only the file handle without re-parsing.

        Safe to call multiple times — no-op if already open.

        !!! tip "Returns self"
            This method returns `self`, so you can chain it directly after the
            constructor if you prefer that style:

            ```python
            bf = BioFile(path).open()
            ```
        """
        with self._lock:
            if self._java_reader is not None and not self._suspended:
                return self  # Already open

            if self._suspended:
                # Fast path: reacquire file handle only
                try:
                    self._java_reader.reopenFile()  # type: ignore[union-attr]
                except Exception:
                    self.destroy()
                    raise
                self._suspended = False
                return self

            # Full initialization (first open, or after context-manager exit)
            r = jimport("loci.formats.ImageReader")()
            r.setFlattenedResolutions(False)

            if self._memoize > 0:
                Memoizer = jimport("loci.formats.Memoizer")
                if BIOFORMATS_MEMO_DIR is not None:
                    r = Memoizer(r, self._memoize, BIOFORMATS_MEMO_DIR)
                else:
                    r = Memoizer(r, self._memoize)

            if self._meta:
                r.setMetadataStore(self._create_ome_meta())
            if self._original_meta:
                r.setOriginalMetadataPopulated(True)

            if self._options:
                mo = jimport("loci.formats.in_.DynamicMetadataOptions")()
                for name, value in self._options.items():
                    mo.set(name, str(value))
                r.setMetadataOptions(mo)

            try:
                r.setId(self._path)
                core_meta = self._get_core_metadata(r)
            except Exception:  # pragma: no cover
                with suppress(Exception):
                    r.close()
                raise

            self._java_reader = r
            self._core_meta_list = core_meta
            self._finalizer = weakref.finalize(self, _close_java_reader, r)
        return self

    def close(self) -> None:
        """Close file handles while preserving reader state for fast reopen.

        Releases the underlying file handle via Java `close(fileOnly=true)`
        but keeps all parsed metadata and reader state in memory. A subsequent
        `open()` call will cheaply reacquire the file handle.

        Metadata remains accessible via `core_metadata()` and `len()`
        while the file is closed.

        Safe to call multiple times — no-op if already closed.
        """
        with self._lock:
            if self._java_reader is not None and not self._suspended:
                self._java_reader.close(True)  # fileOnly=True
                self._suspended = True

    def destroy(self) -> None:
        """Full cleanup — release all Java resources and cached state.

        Releases the Java reader and all parsed metadata, returning to
        UNINITIALIZED. `open()` can be called again but will require full
        re-initialization (slow path). The GC finalizer calls this
        automatically if not called explicitly.

        Safe to call multiple times or from any state.
        """
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()
                self._finalizer = None
            self._java_reader = None
            self._core_meta_list = None
            self._suspended = False

    def as_array(self, series: int = 0, resolution: int = 0) -> LazyBioArray:
        """Return a lazy numpy-compatible array that reads data on-demand.

        The returned array behaves like a numpy array but reads data from disk
        only when indexed. Use it just like a numpy array - any indexing operation
        will read only the requested planes or sub-regions, not the entire dataset.

        Supports integer and slice indexing on all dimensions. The array also
        implements the `__array__()` protocol for seamless numpy integration.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0

        Returns
        -------
        LazyBioArray
            Lazy array in (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb) format

        Examples
        --------
        Index like a numpy array - only reads what you request:

        >>> with BioFile("image.nd2") as bf:
        ...     arr = bf.as_array()  # No data read yet
        ...
        ...     # Read single plane (t=0, c=0, z=2)
        ...     plane = arr[0, 0, 2]  # Only this plane read from disk
        ...
        ...     # Read all timepoints for one channel/z
        ...     timeseries = arr[:, 0, 2]  # Reads T planes
        ...
        ...     # Read sub-region across entire volume
        ...     roi = arr[:, :, :, 100:200, 50:150]  # Reads 100x50 sub-regions
        ...
        ...     # Materialize entire dataset (two equivalent ways)
        ...     full_data = arr[:]  # Using slice notation
        ...     full_data = np.array(arr)  # Using numpy conversion

        Notes
        -----
        BioFile must remain open while using the array. Multiple arrays can
        coexist, each reading from their own series independently.

        Planes >2GB automatically use tiled reading (transparent, ~20% slower).
        """
        if self.closed:
            raise RuntimeError("File not open - call open() first")

        from bffile._lazy_array import LazyBioArray

        return LazyBioArray(self, series, resolution)

    def to_dask(
        self,
        series: int = 0,
        resolution: int = 0,
        chunks: str | tuple = "auto",
        tile_size: tuple[int, int] | str | None = None,
    ) -> dask.array.Array:
        """Create dask array for lazy computation on Bio-Formats data.

        Returns a dask array in TCZYX[r] order that wraps a
        [`LazyBioArray`][bffile.LazyBioArray]. Uses single-threaded scheduler by default
        for Bio-Formats thread safety.

        Parameters
        ----------
        series : int, optional
            Series index to read from, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0
        chunks : str or tuple, default "auto"
            Chunk specification. Examples:
            - "auto": Let dask decide (default)
            - (1, 1, 1, -1, -1): Full Y,X planes per T,C,Z
            - (1, 1, 1, 512, 512): 512x512 tiles
            Mutually exclusive with tile_size.
        tile_size : tuple[int, int] or "auto", optional
            Tile-based chunking for Y,X dimensions (T,C,Z get chunks of 1).
            - (512, 512): Use 512x512 tiles
            - "auto": Query Bio-Formats optimal tile size
            Mutually exclusive with chunks.

        Returns
        -------
        dask.array.Array
            Dask array that reads data on-demand. Shape is (T, C, Z, Y, X) or
            (T, C, Z, Y, X, rgb) for RGB images.

        Raises
        ------
        ValueError
            If both chunks and tile_size are specified

        Examples
        --------
        >>> with BioFile("image.nd2") as bf:
        ...     darr = bf.to_dask(chunks=(1, 1, 1, -1, -1))
        ...     result = darr.mean(axis=2).compute()  # Z-projection

        Notes
        -----
        - BioFile must remain open during computation
        - Uses synchronous scheduler by default (required for thread safety)
        """
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Dask is required for to_dask(). "
                "Please install with `pip install bffile[dask]`"
            ) from e

        # Validate mutually exclusive parameters
        if tile_size is not None and chunks != "auto":
            raise ValueError(
                "chunks and tile_size are mutually exclusive. "
                "When using tile_size, leave chunks as 'auto' (default)."
            )

        # Compute chunks from tile_size if provided
        if tile_size is not None:
            # Validate tile_size format
            if tile_size == "auto":
                # Query Bio-Formats for optimal tile size
                reader = self._ensure_java_reader()
                reader.setSeries(series)
                reader.setResolution(resolution)
                tile_size = (
                    reader.getOptimalTileHeight(),
                    reader.getOptimalTileWidth(),
                )
            elif not (
                isinstance(tile_size, tuple)
                and len(tile_size) == 2
                and all(isinstance(x, int) for x in tile_size)
            ):
                raise ValueError(
                    f"tile_size must be a tuple of two integers or 'auto', "
                    f"got {tile_size}"
                )

            # Compute chunks based on tile size
            meta = self.core_metadata(series, resolution)
            nt, nc, nz, ny, nx, nrgb = meta.shape
            chunks = _utils.get_dask_tile_chunks(nt, nc, nz, ny, nx, tile_size)
            if nrgb > 1:
                chunks = (*chunks, nrgb)  # type: ignore[assignment]

        lazy_arr = self.as_array(series=series, resolution=resolution)
        return da.from_array(lazy_arr, chunks=chunks)  # type: ignore

    @property
    def closed(self) -> bool:
        """Whether the underlying file handles are currently closed."""
        return self._java_reader is None or self._suspended

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        return self._path

    @property
    def ome_xml(self) -> str:
        """Return plain OME XML string."""
        reader = self._ensure_java_reader()
        if store := reader.getMetadataStore():
            try:
                # get metadatastore can return various types of objects,
                # only the OME pyramidal metadata has dumpXML method,
                # (but it's also the most common case here and only useful one)
                # so just warn on error and return empty string.
                return str(store.dumpXML())  # pyright: ignore
            except Exception as e:
                warnings.warn(
                    f"Failed to retrieve OME XML: {e}", RuntimeWarning, stacklevel=2
                )
        return ""

    @property
    def ome_metadata(self) -> OME:
        """Return [`ome_types.OME`][] object parsed from OME XML."""
        if not (omx_xml := self.ome_xml):  # pragma: no cover (not sure if possible)
            return OME()
        xml = _utils.clean_ome_xml_for_known_issues(omx_xml)
        return OME.from_xml(xml)

    def __enter__(self) -> Self:
        """Enter context manager - ensures file is open."""
        self.open()  # Idempotent, so safe to call
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit context manager — full resource cleanup, including the Java reader."""
        self.destroy()

    def __len__(self) -> int:
        """Return the number of series in the file."""
        if self._core_meta_list is None:
            raise RuntimeError("File not open - call open() first")
        return len(self._core_meta_list)

    def series_count(self) -> int:
        """Return the number of series in the file (same as [`__len__`][bffile.BioFile.__len__])."""  # noqa: E501
        return len(self)

    @overload
    def __getitem__(self, index: int) -> Series: ...
    @overload
    def __getitem__(self, index: slice) -> list[Series]: ...
    def __getitem__(self, index: int | slice) -> Series | list[Series]:
        """Return a [`Series`][bffile.Series] proxy for the given index.

        Parameters
        ----------
        index : int | slice
            Series index or slice of series indices to retrieve
        """
        if isinstance(index, int):
            return self._get_series(index)
        elif isinstance(index, slice):
            return [self._get_series(i) for i in range(*index.indices(len(self)))]
        raise TypeError(f"Invalid index type: {type(index)}")  # pragma: no cover

    def _get_series(self, index: int) -> Series:
        """Internal method to get a Series with index validation."""
        n = len(self)  # also validates open state
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Series index {index} out of range (file has {n} series)")
        from bffile._series import Series

        return Series(self, index)

    def used_files(self, *, metadata_only: bool = False) -> list[str]:
        """Return complete list of files needed to open this dataset.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only return files that do not contain pixel data (e.g., metadata,
            companion files, etc...), by default `False`.
        """
        return [
            str(x) for x in self._ensure_java_reader().getUsedFiles(metadata_only) or ()
        ]

    def lookup_table(self, series: int = 0) -> np.ndarray | None:
        """Return the color lookup table for an indexed-color series.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0.

        Returns
        -------
        np.ndarray | None
            Array of shape `(N, 3)` with RGB values for each pixel index,
            or None if the series is not indexed.

        Examples
        --------

        To convert this object to a [`cmap.Colormap`][]:

        ``python
        import cmap
        from bffile import BioFile

        with BioFile("indexed_image.ome.tiff") as bf:
            lut = bf.lookup_table()
            if lut is not None:
                colormap = cmap.Colormap(lut / lut.max())  # Normalize to [0, 1]
        ``
        """
        # `is_indexed` is a bit of a misnomer here (but bioformats uses it)
        # it really means "a LUT exists", not "the image is palette-based"
        #   - True palette images (GIF, indexed PNG) → is_indexed=True, is_rgb=False
        #   - Fluorescence with display LUT (ND2, CZI) → is_indexed=True, is_rgb=False
        #   - Plain grayscale (basic TIFF, no LUT) → is_indexed=False, is_rgb=False
        #   - RGB images → is_indexed=False, is_rgb=True
        if not self.core_metadata(series).is_indexed:
            return None

        reader = self._ensure_java_reader()
        reader.setSeries(series)
        # Many readers require at least one openBytes call before LUT is available
        reader.openBytes(0, 0, 0, 1, 1)
        if (lut := reader.get16BitLookupTable()) is not None:
            return np.asarray(lut, dtype=np.uint16).T
        if (lut := reader.get8BitLookupTable()) is not None:
            return np.asarray(lut, dtype=np.uint8).T
        return None

    def __iter__(self) -> Iterator[Series]:
        """Iterate over all series in the file."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        name = Path(self._path).name
        if self.closed:
            return f"BioFile('{name}', closed)"
        return f"BioFile('{name}', {len(self)} series)"

    def read_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice | None = None,
        x: slice | None = None,
        series: int = 0,
        resolution: int = 0,
        buffer: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read a single plane or sub-region directly from Bio-Formats.

        Low-level method wrapping Bio-Formats' `openBytes()` API. Provides
        fine-grained control for reading specific planes or rectangular
        sub-regions. Most users should use `as_array()` or `to_dask()` instead.

        **Not thread-safe.** Create separate BioFile instances per thread.

        Parameters
        ----------
        t : int, optional
            Time index, by default 0
        c : int, optional
            Channel index, by default 0
        z : int, optional
            Z-slice index, by default 0
        y : slice, optional
            Y-axis slice (default: full height). Example: `slice(100, 200)`
        x : slice, optional
            X-axis slice (default: full width). Example: `slice(50, 150)`
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0
        buffer : np.ndarray, optional
            Pre-allocated buffer for efficient reuse in loops

        Returns
        -------
        np.ndarray
            Shape (height, width) for grayscale or (height, width, rgb) for RGB

        Examples
        --------
        >>> with BioFile("image.nd2") as bf:
        ...     plane = bf.read_plane(t=0, c=1, z=5)
        ...     roi = bf.read_plane(y=slice(200, 300), x=slice(200, 300))

        See Also
        --------
        as_array : Create a numpy-compatible lazy array
        to_dask : Create a dask array for lazy loading
        """
        reader = self._ensure_java_reader()
        reader.setSeries(series)
        reader.setResolution(resolution)

        # Get metadata for this series/resolution
        meta = self.core_metadata(series, resolution)
        shape = meta.shape

        # Handle default slices
        y = y if y is not None else slice(0, shape.y)
        x = x if x is not None else slice(0, shape.x)

        # Call optimized internal method
        im = self._read_plane(reader, meta, t, c, z, y, x)

        # If buffer provided, copy into it (for reuse in loops)
        if buffer is not None:
            buffer[:] = im
            return buffer

        return im

    def _read_plane(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        t: int,
        c: int,
        z: int,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """Fast plane reading for hot path (internal use only).


        This method skips all validation, metadata lookups, and series/resolution
        setting, assuming they've been done once before entering a tight loop.

        It *does*, however, dispatch to tiled or direct read based on plane size.
        (Note: users have full power to control tiling via slicing into LazyBioArray,
        or by using to_dask()... this is just a safety net for requests that would
        exceed Java limits.)
        """
        shape = meta.shape
        y_start, y_stop, _ = y.indices(shape.y)
        x_start, x_stop, _ = x.indices(shape.x)

        height = y_stop - y_start
        width = x_stop - x_start
        plane_bytes = height * width * meta.dtype.itemsize * meta.shape.rgb

        if plane_bytes > MAX_JAVA_ARRAY_SIZE:
            return self._read_plane_tiled(
                reader, meta, t, c, z, y_start, x_start, height, width
            )
        return self._read_plane_direct(
            reader, meta, t, c, z, y_start, x_start, height, width
        )

    def _read_plane_direct(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        t: int,
        c: int,
        z: int,
        y_start: int,
        x_start: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Read plane directly (fast path for <2GB planes)."""
        shape = meta.shape
        idx = reader.getIndex(z, c, t)

        java_buffer = reader.openBytes(idx, x_start, y_start, width, height)
        im = np.frombuffer(memoryview(java_buffer), meta.dtype)  # type: ignore

        # Reshape
        if shape.rgb > 1:
            if meta.is_interleaved:
                im.shape = (height, width, shape.rgb)
            else:
                im.shape = (shape.rgb, height, width)
                im = np.transpose(im, (1, 2, 0))
        else:
            im.shape = (height, width)

        return im

    def _calculate_tile_height(self, meta: CoreMetadata, region_width: int) -> int:
        """Calculate max rows per tile respecting Java array limit and heap space.

        Uses full-width rows (no X tiling) to minimize openBytes() calls.
        """
        row_bytes = region_width * meta.dtype.itemsize * meta.shape.rgb

        # Constraint 1: Java's max byte array size
        tile_height = MAX_JAVA_ARRAY_SIZE // row_bytes

        # Constraint 2: Available heap (with 80% safety margin)
        rt = jimport("java.lang.Runtime").getRuntime()
        available_heap = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
        max_heap_rows = int(available_heap * 0.8) // row_bytes

        return max(1, min(tile_height, max_heap_rows))

    def _read_plane_tiled(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        t: int,
        c: int,
        z: int,
        y_start: int,
        x_start: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Read large plane via tiling to avoid 2GB Java array limit.

        Key insight: openBytes() dominates time (~98%), so minimize tile count.
        Strategy: Reuse one large buffer, read full-width rows, copy to output.
        """
        shape = meta.shape

        # Preallocate output
        output_shape = (height, width, shape.rgb) if shape.rgb > 1 else (height, width)
        output = np.empty(output_shape, dtype=meta.dtype)

        # Calculate tile size
        tile_height = self._calculate_tile_height(meta, width)
        row_bytes = width * meta.dtype.itemsize * meta.shape.rgb

        # Allocate buffer with fallback for OOM
        # Key lesson: Heap size often limits us before theoretical 2GB limit
        tile_buffer = None
        min_tile_height = max(1, height // 100)
        OutOfMemoryError = jimport("java.lang.OutOfMemoryError")
        while tile_buffer is None and tile_height >= min_tile_height:
            try:
                tile_buffer = jpype.JArray(jpype.JByte)(tile_height * row_bytes)  # pyright: ignore[reportCallIssue]
            except OutOfMemoryError as e:
                tile_height //= 2
                if tile_height < min_tile_height:
                    gb = tile_height * row_bytes / 1024**3
                    raise MemoryError(
                        f"Cannot allocate {gb:.2f} GB tile buffer. "
                        f"Set JAVA_TOOL_OPTIONS='-Xmx8g' to increase heap. Or further "
                        "reduce tile size by setting the environment variable "
                        "BIOFORMATS_MAX_JAVA_BYTES to a smaller value."
                    ) from e

        plane_idx = reader.getIndex(z, c, t)

        # Read tiles
        y_offset = 0
        for y0 in range(0, height, tile_height):
            h = min(tile_height, height - y0)

            reader.openBytes(plane_idx, tile_buffer, x_start, y_start + y0, width, h)  # pyright: ignore[reportArgumentType]

            # View tile data (count is elements, not bytes)
            tile_data = np.frombuffer(
                memoryview(tile_buffer),  # pyright: ignore[reportArgumentType]
                dtype=meta.dtype,
                count=h * width * shape.rgb,
            )

            # Copy to output
            if shape.rgb > 1:
                if meta.is_interleaved:
                    output[y_offset : y_offset + h].ravel()[:] = tile_data
                else:
                    # Non-interleaved needs transpose
                    tile = tile_data.reshape(shape.rgb, h, width).transpose(1, 2, 0)
                    output[y_offset : y_offset + h] = tile
            else:
                output[y_offset : y_offset + h].ravel()[:] = tile_data

            y_offset += h

        return output

    _service: ClassVar[Any] = None

    @classmethod
    def _create_ome_meta(cls) -> Any:
        """Create an OMEXMLMetadata object to populate."""
        if cls._service is None:
            ServiceFactory = jimport("loci.common.services.ServiceFactory")
            OMEXMLService = jimport("loci.formats.services.OMEXMLService")

            factory = ServiceFactory()
            cls._service = factory.getInstance(OMEXMLService)

        return cls._service.createOMEXMLMetadata()

    @staticmethod
    def bioformats_version() -> str:
        """Get the version of Bio-Formats."""
        Version = jimport("loci.formats.FormatTools")
        return str(getattr(Version, "VERSION", "unknown"))

    @staticmethod
    def bioformats_maven_coordinate() -> str:
        """Return the Maven coordinate used to load Bio-Formats.

        This was either provided via the `BIOFORMATS_VERSION` environment variable, or
        is the default value, in format "groupId:artifactId:version",
        See <https://mvnrepository.com/artifact/ome> for available versions.
        """
        from ._java_stuff import MAVEN_COORDINATE

        return MAVEN_COORDINATE

    @staticmethod
    @cache
    def list_supported_suffixes() -> set[str]:
        """List all file suffixes supported by the available readers."""
        reader = jimport("loci.formats.ImageReader")()
        return {str(x) for x in reader.getSuffixes()}

    @staticmethod
    @cache
    def list_available_readers() -> list[ReaderInfo]:
        """List all available Bio-Formats readers.

        Returns
        -------
        list[ReaderInfo]
            Information about each available reader, including:

            - format: human-readable format name (e.g., "Nikon ND2")
            - suffixes: supported file extensions (e.g., ("nd2", "jp2"))
            - class_name: full Java class name (e.g., "ND2Reader")
            - is_gpl: whether this reader requires GPL license (True) or is BSD (False)
        """
        ImageReader = jimport("loci.formats.ImageReader")
        temp_reader = ImageReader()
        try:
            formats = []
            for reader in temp_reader.getReaders():
                reader_cls = cast("java.lang.Class", reader.getClass())  # type: ignore
                class_name = str(reader_cls.getName()).removeprefix("loci.formats.in.")

                # Detect license from JAR file name
                # GPL readers come from formats-gpl-X.X.X.jar
                # BSD readers come from formats-bsd-X.X.X.jar
                is_gpl = True
                with suppress(Exception):
                    protection_domain = reader_cls.getProtectionDomain()
                    if (code_source := protection_domain.getCodeSource()) is not None:
                        location = str(code_source.getLocation())
                        is_gpl = "formats-gpl-" in location.split("/")[-1]

                formats.append(
                    ReaderInfo(
                        format=str(reader.getFormat()),
                        suffixes=tuple(str(s) for s in reader.getSuffixes()),
                        class_name=class_name,
                        is_gpl=is_gpl,
                    )
                )

            return formats
        finally:
            temp_reader.close()


def _close_java_reader(java_reader: IFormatReader | None) -> None:
    """Close a Java reader if JVM is still running.

    Used as weakref finalizer for last-resort cleanup. This can ONLY close
    the Java file handle - it cannot access Python instance state because
    it's called after the BioFile instance is garbage collected.

    For explicit cleanup, use the BioFile.close() method instead.
    """
    # Only attempt close during normal operation (not shutdown)
    if java_reader is None or sys.is_finalizing() or not jpype.isJVMStarted():
        return  # pragma: no cover
    with suppress(Exception):
        java_reader.close()
