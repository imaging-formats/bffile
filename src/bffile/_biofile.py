from __future__ import annotations

import os
import sys
import warnings
import weakref
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar

import jpype
import numpy as np
from ome_types import OME
from typing_extensions import Self

from bffile._core_metadata import CoreMetadata, OMEShape

from . import _utils
from ._jimports import jimport

if TYPE_CHECKING:
    from loci.formats import IFormatReader
    from resource_backed_dask_array import ResourceBackedDaskArray


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
        path to file
    series : int, optional
        the image series to read, by default 0
    meta : bool, optional
        whether to get metadata as well, by default True
    original_meta : bool, optional
        whether to also retrieve the proprietary metadata as structured annotations in
        the OME output, by default False
    memoize : bool or int, optional
        threshold (in milliseconds) for memoizing the reader. If the the time
        required to call `reader.setId()` is larger than this number, the initialized
        reader (including all reader wrappers) will be cached in a memo file, reducing
        time to load the file on future reads.  By default, this results in a hidden
        `.bfmemo` file in the same directory as the file. The `BIOFORMATS_MEMO_DIR`
        environment can be used to change the memo file directory.
        Set `memoize` to greater than 0 to turn on memoization. by default it's off.
        https://downloads.openmicroscopy.org/bio-formats/latest/api/loci/formats/Memoizer.html
    options : Dict[str, bool], optional
        A mapping of option-name -> bool specifying additional reader-specific options.
        see: https://docs.openmicroscopy.org/bio-formats/latest/formats/options.html
        For example: to turn off chunkmap table reading for ND2 files, use
        `options={"nativend2.chunkmap": False}`
    dask_tiles: bool, optional
        Whether to chunk the bioformats dask array by tiles to easily read sub-regions
        with numpy-like array indexing
        Defaults to false and images are read by entire planes
    tile_size: Optional[Tuple[int, int]]
        Tuple that sets the tile size of y and x axis, respectively
        By default, it will use optimal values computed by bioformats itself
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        series: int = 0,
        meta: bool = True,
        original_meta: bool = False,
        memoize: int | bool = 0,
        options: dict[str, bool] | None = None,
        dask_tiles: bool = False,
        tile_size: tuple[int, int] | None = None,
    ):
        self._path = str(Path(path).expanduser().absolute())
        self._current_scene_index = series
        self._lock = RLock()
        self.dask_tiles = dask_tiles

        if tile_size is not None:
            if len(tile_size) != 2:
                raise ValueError(f"tile_size must be length 2, got {len(tile_size)}")
            if not all(isinstance(x, int) for x in tile_size):
                raise ValueError(f"tile_size must be integers, got {tile_size}")
            tile_size = tuple(tile_size)  # type: ignore[assignment]

        self._tile_size_override = tile_size
        self._meta = meta
        self._original_meta = original_meta
        self._memoize = memoize
        self._options = options

        # Reader and finalizer created in open()
        self._java_reader: IFormatReader | None = None
        self._core_meta_list: list[CoreMetadata] | None = None
        self._finalizer: weakref.finalize | None = None

    def set_series(self, series: int = 0) -> None:
        """Set the current image series.

        Parameters
        ----------
        series : int
            Series index to select

        Raises
        ------
        RuntimeError
            If file is not open
        """
        with self._lock:
            if self.closed:
                raise RuntimeError("Cannot set series on closed file")
            self.java_reader().setSeries(series)
            self._current_scene_index = series

    def java_reader(self) -> IFormatReader:
        """Return the native reader object.

        Raises
        ------
        RuntimeError
            If file is not open
        """
        if self.closed:  # Uses finalizer.alive under the hood
            raise RuntimeError("File not open - call open() first")
        if self._java_reader is None:  # Should never happen, but type safety
            raise RuntimeError("Internal error: reader not initialized")
        return self._java_reader

    @property
    def core_meta(self) -> CoreMetadata:
        """Get metadata for current series.

        Raises
        ------
        RuntimeError
            If file is not open
        """
        if self._core_meta_list is None:
            raise RuntimeError("File not open - call open() first")
        return self._core_meta_list[self._current_scene_index]

    @property
    def shape(self) -> OMEShape:
        return self.core_meta.shape

    @property
    def dtype(self) -> np.dtype:
        return self.core_meta.dtype

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
            self._java_reader = jimport("loci.formats.ImageReader")()

            # Wrap with Memoizer if requested
            # Note: Memoizer MUST wrap before setMetadataStore
            if self._memoize > 0:
                Memoizer = jimport("loci.formats.Memoizer")
                if BIOFORMATS_MEMO_DIR is not None:
                    self._java_reader = Memoizer(
                        self._java_reader, self._memoize, BIOFORMATS_MEMO_DIR
                    )
                else:
                    self._java_reader = Memoizer(self._java_reader, self._memoize)

            # Configure reader
            if self._meta:
                self._java_reader.setMetadataStore(self._create_ome_meta())
            if self._original_meta:
                self._java_reader.setOriginalMetadataPopulated(True)

            if self._options:
                DynamicMetadataOptions = jimport(
                    "loci.formats.in_.DynamicMetadataOptions"
                )
                mo = DynamicMetadataOptions()
                for name, value in self._options.items():
                    mo.set(name, str(value))
                self._java_reader.setMetadataOptions(mo)

            # Open file - this is the critical operation that can fail
            try:
                self._java_reader.setId(self._path)

                # Cache metadata
                self._core_meta_list = [
                    CoreMetadata.from_java(x)
                    for x in self._java_reader.getCoreMetadataList()
                ]

                # Set the series specified in __init__
                # Note: set_series() acquires lock, but we already have it
                # So we call the Java method directly here
                self._java_reader.setSeries(self._current_scene_index)

                # Setup tile size if needed
                if self.dask_tiles:
                    if self._tile_size_override is None:
                        self.tile_size: tuple[int, int] = (
                            self._java_reader.getOptimalTileHeight(),
                            self._java_reader.getOptimalTileWidth(),
                        )
                    else:
                        self.tile_size = self._tile_size_override

                # The finalizer's alive state is now the source of truth for open/closed
                self._finalizer = weakref.finalize(
                    self, _close_java_reader, self._java_reader
                )

            except Exception:
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

    def to_numpy(self, series: int | None = None) -> np.ndarray:
        """Create numpy array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        Parameters
        ----------
        series : int, optional
            The series index to retrieve, by default None
        """
        if self._java_reader is None:
            raise RuntimeError("File not open - call open() first")

        if series is not None:
            self._java_reader.setSeries(series)

        nt, nc, nz, ny, nx, nrgb = self.core_meta.shape

        # Create output array with appropriate shape
        if nrgb > 1:
            output = np.empty((nt, nc, nz, ny, nx, nrgb), dtype=self.core_meta.dtype)
        else:
            output = np.empty((nt, nc, nz, ny, nx), dtype=self.core_meta.dtype)

        # Fill in each plane
        for t in range(nt):
            for c in range(nc):
                for z in range(nz):
                    plane = self._get_plane(t, c, z)
                    output[t, c, z] = plane

        return output

    def to_dask(self, series: int | None = None) -> ResourceBackedDaskArray:
        """Create dask array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        The returned object is a `ResourceBackedDaskArray`, which is a wrapper on
        a dask array that ensures the file is open when actually reading (computing)
        a chunk.  It has all the methods and behavior of a dask array.
        See: https://github.com/tlambert03/resource-backed-dask-array

        Returns
        -------
        ResourceBackedDaskArray
        """
        try:
            import dask.array as da
            from resource_backed_dask_array import resource_backed_dask_array
        except ImportError as e:
            raise ImportError(
                "Dask and resource-backed-dask-array are required for to_dask(). "
                "Please install with `pip install bffile[dask]`"
            ) from e

        if self._java_reader is None:
            raise RuntimeError("File not open - call open() first")

        if series is not None:
            self._java_reader.setSeries(series)

        nt, nc, nz, ny, nx, nrgb = self.core_meta.shape

        if self.dask_tiles:
            chunks = _utils.get_dask_tile_chunks(nt, nc, nz, ny, nx, self.tile_size)
        else:
            chunks = ((1,) * nt, (1,) * nc, (1,) * nz, (ny,), (nx,))

        if nrgb > 1:
            chunks = (*chunks, nrgb)  # type: ignore[assignment]

        arr = da.map_blocks(
            self._dask_chunk,
            chunks=chunks,
            dtype=self.core_meta.dtype,
        )
        return resource_backed_dask_array(arr, self)

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
        if not (omx_xml := self.ome_xml):
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

    def _get_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice | None = None,
        x: slice | None = None,
    ) -> np.ndarray:
        """Load bytes from a single plane.

        The file must be open before calling this method.

        Parameters
        ----------
        t : int, optional
            the time index, by default 0
        c : int, optional
            the channel index, by default 0
        z : int, optional
            the z index, by default 0
        y : slice, optional
            a slice object to select a Y subset of the plane, by default: full axis.
        x : slice, optional
            a slice object to select a X subset of the plane, by default: full axis.

        Returns
        -------
        np.ndarray
            array of requested bytes.

        Raises
        ------
        RuntimeError
            If file is not open
        """
        with self._lock:
            # Don't auto-reopen - require explicit open
            if self.closed:
                raise RuntimeError(
                    "Cannot read from closed file. "
                    "Call open() first or use a context manager: "
                    "with BioFile(...) as bf:"
                )

            if self._java_reader is None or self._core_meta_list is None:
                raise RuntimeError(
                    "Metadata not initialized - file may not be properly opened"
                )

            shape = self.core_meta.shape

            y = y if y is not None else slice(0, shape.y)
            x = x if x is not None else slice(0, shape.x)

            # get bytes from bioformats
            idx = self._java_reader.getIndex(z, c, t)
            ystart, ywidth = _utils.slice2width(y, shape.y)
            xstart, xwidth = _utils.slice2width(x, shape.x)
            # read bytes using bioformats
            buffer = self._java_reader.openBytes(idx, xstart, ystart, xwidth, ywidth)
            # convert buffer to numpy array
            im = np.frombuffer(bytes(buffer), self.core_meta.dtype)

            # reshape
            if shape.rgb > 1:
                if self.core_meta.is_interleaved:
                    im.shape = (ywidth, xwidth, shape.rgb)
                else:
                    im.shape = (shape.rgb, ywidth, xwidth)
                    im = np.transpose(im, (1, 2, 0))
            else:
                im.shape = (ywidth, xwidth)

        return im

    def _dask_chunk(self, block_id: tuple[int, ...]) -> np.ndarray:
        """Retrieve `block_id` from array.

        This function is for map_blocks (called in `to_dask`).
        If someone indexes a 5D dask array as `arr[0, 1, 2]`, then 'block_id'
        will be (0, 1, 2, 0, 0)
        """
        # Our convention is that the final dask array is in the order TCZYX, so
        # block_id will be coming in as (T, C, Z, Y, X).
        t, c, z, y, x, *_ = block_id

        if self.dask_tiles:
            *_, ny, nx, _ = self.core_meta.shape
            y_slice = _utils.axis_id_to_slice(y, self.tile_size[0], ny)
            x_slice = _utils.axis_id_to_slice(x, self.tile_size[1], nx)
            im = self._get_plane(t, c, z, y_slice, x_slice)
        else:
            im = self._get_plane(t, c, z)

        return im[np.newaxis, np.newaxis, np.newaxis]

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


def _close_java_reader(java_reader: IFormatReader | None) -> None:
    """Close a Java reader if JVM is still running.

    Used as weakref finalizer for last-resort cleanup. This can ONLY close
    the Java file handle - it cannot access Python instance state because
    it's called after the BioFile instance is garbage collected.

    For explicit cleanup, use the BioFile.close() method instead.
    """
    if java_reader is None:
        return
    # Only attempt close during normal operation (not shutdown)
    if not sys.is_finalizing() and jpype.isJVMStarted():
        with suppress(Exception):
            java_reader.close()
