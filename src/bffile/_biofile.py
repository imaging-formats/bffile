from __future__ import annotations

import os
import sys
import warnings
import weakref
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar, cast

import jpype
import numpy as np
from ome_types import OME
from typing_extensions import Self

from bffile._core_metadata import CoreMetadata

from . import _utils
from ._jimports import jimport

if TYPE_CHECKING:
    import dask.array
    import java.lang
    from loci.formats import IFormatReader

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


class BioFile:
    """Read image and metadata from file supported by Bioformats.

    BioFile instances must be explicitly opened before use, either by:
    1. Using a context manager: `with BioFile(path) as bf: ...`
    2. Explicitly calling `open()` and `close()`:
       `bf = BioFile(path); bf.open(); ...; bf.close()`

    The recommended pattern is to use the context manager, which automatically
    handles opening and closing the file.

    BioFile instances are not thread-safe. Create separate instances per thread.

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
        self._path = str(Path(path).expanduser().absolute())
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

    def java_reader(self) -> IFormatReader:
        """Return the native reader object.

        Raises
        ------
        RuntimeError
            If file is not open
        """
        if self._java_reader is None:
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

    def core_meta(self, series: int = 0, resolution: int = 0) -> CoreMetadata:
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

    def open(self) -> None:
        """Open file and initialize reader.

        Safe to call multiple times - will only initialize once.
        If file is already open, this is a no-op.
        """
        with self._lock:
            # If already open, nothing to do
            if self._java_reader is not None:
                return

            # Create reader
            self._java_reader = reader = jimport("loci.formats.ImageReader")()

            # Use non-flattened resolution mode for cleaner API
            # This allows us to use setSeries(s) + setResolution(r) instead of
            # treating each resolution as a separate series
            reader.setFlattenedResolutions(False)

            # Wrap with Memoizer if requested
            # Note: Memoizer MUST wrap before setMetadataStore
            if self._memoize > 0:
                Memoizer = jimport("loci.formats.Memoizer")
                if BIOFORMATS_MEMO_DIR is not None:
                    reader = Memoizer(reader, self._memoize, BIOFORMATS_MEMO_DIR)
                else:
                    reader = Memoizer(reader, self._memoize)

            # Configure reader
            if self._meta:
                reader.setMetadataStore(self._create_ome_meta())
            if self._original_meta:
                reader.setOriginalMetadataPopulated(True)

            if self._options:
                mo = jimport("loci.formats.in_.DynamicMetadataOptions")()
                for name, value in self._options.items():
                    mo.set(name, str(value))
                reader.setMetadataOptions(mo)

            # Open file - this is the critical operation that can fail
            try:
                reader.setId(self._path)
                self._core_meta_list = self._get_core_metadata(reader)
                # The finalizer's alive state is now the source of truth for open/closed
                self._finalizer = weakref.finalize(self, _close_java_reader, reader)

            except Exception:  # pragma: no cover
                self.close()
                raise

    def close(self) -> None:
        """Close file and release resources.

        Safe to call multiple times - will only close once.
        After closing, the BioFile instance can be reopened by calling open().
        """
        with self._lock:
            # Call the finalizer if it exists
            if self._finalizer is not None:
                self._finalizer()
                self._finalizer = None

            # Clear cached references
            self._java_reader = None
            self._core_meta_list = None

    def as_array(self, series: int = 0, resolution: int = 0) -> LazyBioArray:
        """Return a lazy numpy-compatible array that reads data on-demand.

        The returned array supports numpy-style indexing and implements the
        `__array__()` protocol for seamless numpy integration. To cast it to a numpy
        array (which materializes *all* data for the given series), use `np.array()` on
        it.

        Parameters
        ----------
        series : int, optional
            The series index to retrieve, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0

        Returns
        -------
        LazyBioArray
            Lazy array that reads from Bio-Formats on-demand

        Examples
        --------
        >>> with BioFile("image.nd2") as bf:
        ...     arr = bf.as_array(series=0)  # No data read yet
        ...     plane = arr[0, 0, 2]  # Reads only this plane
        ...     roi = arr[0, 0, :, 100:200, 50:150]  # Reads sub-XY-regions across all Z
        ...
        ...     # Convert to numpy when needed
        ...     full_data = np.array(arr)  # Reads all data

        Notes
        -----
        - The BioFile instance must remain open while using the lazy array
        - Multiple LazyBioArray instances can safely coexist, each reading
          from their own series independently
        - Resolution parameter is included for future compatibility but currently
          only resolution 0 is fully supported
        """
        if self._java_reader is None:
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
        """Create dask array for the specified series and resolution.

        This method wraps the LazyBioArray in a dask array, enabling lazy
        computation workflows. The dask array uses single-threaded execution
        by default (required for Bio-Formats thread safety).

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        The dask array wraps a LazyBioArray, which reads data on-demand from
        Bio-Formats. The BioFile instance must remain open while the dask array
        is being computed.

        Parameters
        ----------
        series : int, optional
            Series index to read from, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0
        chunks : str or tuple, default "auto"
            Chunk specification for dask array. Common values:
            - "auto": Let dask decide (default)
            - (1, 1, 1, -1, -1): Each T, C, Z separate, full Y, X planes
            - (1, 1, 1, 512, 512): Tile into 512x512 chunks

            **Mutually exclusive with tile_size.** If tile_size is provided,
            chunks must be "auto" (default).
        tile_size : tuple[int, int] or "auto", optional
            Alternative way to specify chunking for tile-based access patterns.
            If provided, chunks the Y and X dimensions into tiles of the specified
            size, with each T, C, Z as separate chunks.

            - (512, 512): Use 512x512 tiles for Y and X dimensions
            - "auto": Query Bio-Formats for optimal tile size
            - None: Use chunks parameter (default)

            **Mutually exclusive with chunks.** If tile_size is provided,
            chunks must be "auto" (default).

        Returns
        -------
        dask.array.Array
            Returns a standard dask array that reads from the LazyBioArray on-demand.

        Raises
        ------
        ValueError
            If both chunks and tile_size are specified (they are mutually exclusive)

        Examples
        --------
        Basic usage with automatic chunking:

        >>> bf = BioFile("image.nd2")
        >>> darr = bf.to_dask(series=0)  # chunks="auto"

        Explicit chunk specification:

        >>> darr = bf.to_dask(chunks=(1, 1, 1, -1, -1))  # Full Y, X planes

        Tile-based chunking:

        >>> darr = bf.to_dask(tile_size=(512, 512))  # 512x512 tiles
        >>> darr = bf.to_dask(tile_size="auto")  # Bio-Formats optimal tiles

        Execute computation:

        >>> import dask
        >>> with dask.config.set(scheduler="synchronous"):
        ...     result = darr.mean(axis=(1, 2)).compute()
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
                reader = self.java_reader()
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
            meta = self.core_meta(series, resolution)
            nt, nc, nz, ny, nx, nrgb = meta.shape
            chunks = _utils.get_dask_tile_chunks(nt, nc, nz, ny, nx, tile_size)
            if nrgb > 1:
                chunks = (*chunks, nrgb)  # type: ignore[assignment]

        lazy_arr = self.as_array(series=series, resolution=resolution)
        return da.from_array(lazy_arr, chunks=chunks)  # type: ignore

    @property
    def closed(self) -> bool:
        """Whether the underlying file is currently closed."""
        return self._java_reader is None

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        # return self._r.getCurrentFile()
        return self._path

    @property
    def ome_xml(self) -> str:
        """Return OME XML string."""
        reader = self.java_reader()
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
        """Return OME object parsed by ome_types."""
        if not (omx_xml := self.ome_xml):  # pragma: no cover (not sure if possible)
            return OME()
        xml = _utils.clean_ome_xml_for_known_issues(omx_xml)
        return OME.from_xml(xml)

    def __enter__(self) -> Self:
        """Enter context manager - ensures file is open."""
        self.open()  # Idempotent, so safe to call
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit context manager - ensures file is closed."""
        self.close()  # Idempotent, so safe to call

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
        """
        Read a single plane or sub-region directly from Bio-Formats.

        This is the low-level method that directly wraps Bio-Formats' `openBytes()`
        API. It provides maximum control and efficiency for reading specific planes
        or rectangular sub-regions.

        **This method is NOT thread-safe.** For multi-threaded access, create
        separate BioFile instances per thread (recommended) or manage your own
        locking around calls to this method.

        Most users should use higher-level methods like `to_numpy()`, `to_dask()`,
        or `as_array()`. Use this method when you need:

        - Fine-grained control over which planes to read
        - Efficient sub-region reading without loading full planes
        - Custom caching or streaming strategies
        - Direct access to Bio-Formats' coordinate system

        Parameters
        ----------
        t : int, optional
            Time index (default: 0)
        c : int, optional
            Channel index (default: 0)
        z : int, optional
            Z-slice index (default: 0)
        y : slice, optional
            Y-axis slice for sub-region reading (default: full height)
            Example: `slice(100, 200)` reads rows 100-199
        x : slice, optional
            X-axis slice for sub-region reading (default: full width)
            Example: `slice(50, 150)` reads columns 50-149
        series : int, optional
            Series index to read from, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0
        buffer : np.ndarray, optional
            Pre-allocated buffer to read into for efficiency in tight loops.
            If provided, data is copied into this buffer (reusing it across reads).
            Buffer must have correct shape and dtype. Used internally for
            optimization.

        Returns
        -------
        np.ndarray
            2D array of shape (height, width) for grayscale images, or
            3D array of shape (height, width, rgb) for RGB images.
            The array is a view when possible (zero-copy via memoryview).

        Raises
        ------
        RuntimeError
            If file is not open or not properly initialized

        Examples
        --------
        Read a specific plane:

        >>> with BioFile("image.nd2") as bf:
        ...     plane = bf.read_plane(t=0, c=1, z=5)
        ...     print(plane.shape)
        (512, 512)

        Read a sub-region (100x100 pixels at position 200,200):

        >>> with BioFile("image.nd2") as bf:
        ...     roi = bf.read_plane(t=0, c=0, z=0, y=slice(200, 300), x=slice(200, 300))
        ...     print(roi.shape)
        (100, 100)

        Multi-series file (always specify series):

        >>> with BioFile("multi.czi") as bf:
        ...     plane_s0 = bf.read_plane(t=0, c=0, z=0, series=0)
        ...     plane_s1 = bf.read_plane(t=0, c=0, z=0, series=1)

        See Also
        --------
        to_numpy : Load entire dataset into memory (thread-safe)
        to_dask : Create a dask array for lazy loading (thread-safe)
        as_array : Create a numpy-compatible lazy array (thread-safe)

        Notes
        -----
        This method does NOT acquire locks internally. The underlying Bio-Formats
        reader is not thread-safe. For parallel processing, create separate
        BioFile instances per thread rather than sharing a single instance.
        """
        reader = self.java_reader()
        reader.setSeries(series)
        reader.setResolution(resolution)

        # Get metadata for this series/resolution
        meta = self.core_meta(series, resolution)
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
        """
        Fast plane reading for hot path (internal use only).

        This method skips all validation, metadata lookups, and series/resolution
        setting, assuming they've been done once before entering a tight loop.

        Parameters
        ----------
        reader : IFormatReader
            Java reader (already set to correct series/resolution)
        meta : CoreMetadata
            Metadata for current series/resolution (avoid repeated lookups)
        t, c, z : int
            Plane coordinates
        y, x : slice
            Sub-region slices (not None)

        Returns
        -------
        np.ndarray
            Plane data as 2D or 3D array
        """
        shape = meta.shape

        # Get bytes from bioformats
        idx = reader.getIndex(z, c, t)
        ystart, ywidth = _utils.slice2width(y, shape.y)
        xstart, xwidth = _utils.slice2width(x, shape.x)

        # Read bytes using bioformats
        java_buffer = reader.openBytes(idx, xstart, ystart, xwidth, ywidth)
        # Convert buffer to numpy array (zero-copy via memoryview)
        im = np.frombuffer(memoryview(java_buffer), meta.dtype)  # type: ignore

        # Reshape
        if shape.rgb > 1:
            if meta.is_interleaved:
                im.shape = (ywidth, xwidth, shape.rgb)
            else:
                im.shape = (shape.rgb, ywidth, xwidth)
                im = np.transpose(im, (1, 2, 0))
        else:
            im.shape = (ywidth, xwidth)

        return im

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
