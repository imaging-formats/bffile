from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar

import dask.array as da
import numpy as np
from ome_types import OME
from scyjava import jimport
from typing_extensions import Self

from . import _utils
from ._java_stuff import hide_memoization_warning, pixtype2dtype, redirect_java_logging

if TYPE_CHECKING:
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

    BioFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    BioFile instances are not thread-safe.

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
        redirect_java_logging()
        ImageReader = jimport("loci.formats.ImageReader")

        self._path = str(Path(path).expanduser().absolute())
        self._r = ImageReader()

        # memoize to save time on later re-openings of the same file.
        # Note: Memoizer must wrap the reader BEFORE setMetadataStore is called
        if memoize > 0:
            Memoizer = jimport("loci.formats.Memoizer")
            hide_memoization_warning()
            if BIOFORMATS_MEMO_DIR is not None:
                self._r = Memoizer(self._r, memoize, BIOFORMATS_MEMO_DIR)
            else:
                self._r = Memoizer(self._r, memoize)

        if meta:
            self._r.setMetadataStore(self._create_ome_meta())
        if original_meta:
            self._r.setOriginalMetadataPopulated(True)

        if options:
            DynamicMetadataOptions = jimport("loci.formats.in_.DynamicMetadataOptions")
            mo = DynamicMetadataOptions()
            for name, value in options.items():
                mo.set(name, str(value))
            self._r.setMetadataOptions(mo)

        self._current_scene_index = series
        self._is_open = False
        self.open()
        self._lock = Lock()
        self.set_series(series)

        self.dask_tiles = dask_tiles
        if self.dask_tiles:
            if tile_size is None:
                self.tile_size = (
                    self._r.getOptimalTileHeight(),
                    self._r.getOptimalTileWidth(),
                )
            else:
                self.tile_size = tile_size

    def set_series(self, series: int = 0) -> None:
        self._r.setSeries(series)
        self._core_meta = _utils.CoreMeta(
            _utils.OMEShape(
                self._r.getSizeT(),
                self._r.getEffectiveSizeC(),
                self._r.getSizeZ(),
                self._r.getSizeY(),
                self._r.getSizeX(),
                self._r.getRGBChannelCount(),
            ),
            pixtype2dtype(self._r.getPixelType(), self._r.isLittleEndian()),
            self._r.getSeriesCount(),
            self._r.isRGB(),
            self._r.isInterleaved(),
            self._r.getDimensionOrder(),
            self._r.getResolutionCount(),
        )
        self._current_scene_index = series

    @property
    def core_meta(self) -> _utils.CoreMeta:
        return self._core_meta

    @property
    def shape(self) -> _utils.OMEShape:
        return self._core_meta.shape

    def open(self) -> None:
        """Open file."""
        if not self._is_open:
            self._r.setId(self._path)
            self._r.setSeries(self._current_scene_index)
            self._is_open = True

    def close(self) -> None:
        """Close file."""
        with suppress(Exception):
            self._r.close()
            self._is_open = False

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
        # TODO: make going through dask optional
        return np.asarray(self.to_dask(series))

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
        from resource_backed_dask_array import resource_backed_dask_array

        if series is not None:
            self._r.setSeries(series)

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
        """Whether the underlying file is currently open."""
        return not bool(self._r.getCurrentFile())

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        # return self._r.getCurrentFile()
        return self._path

    @property
    def ome_xml(self) -> str:
        """Return OME XML string."""
        if store := self._r.getMetadataStore():
            return str(store.dumpXML())
        return ""

    @property
    def ome_metadata(self) -> OME:
        """Return OME object parsed by ome_types."""
        xml = _utils.clean_ome_xml_for_known_issues(self.ome_xml)
        return OME.from_xml(xml)

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _get_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice | None = None,
        x: slice | None = None,
    ) -> np.ndarray:
        """Load bytes from a single plane.

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
        """
        with self._lock:
            was_open = not self.closed
            if not was_open:
                self.open()

            *_, ny, nx, nrgb = self.core_meta.shape

            y = y if y is not None else slice(0, ny)
            x = x if x is not None else slice(0, nx)

            # get bytes from bioformats
            idx = self._r.getIndex(z, c, t)
            ystart, ywidth = _utils.slice2width(y, ny)
            xstart, xwidth = _utils.slice2width(x, nx)
            # read bytes using bioformats
            buffer = self._r.openBytes(idx, xstart, ystart, xwidth, ywidth)
            # convert buffer to numpy array
            im = np.frombuffer(bytes(buffer), self.core_meta.dtype)

            # reshape
            if nrgb > 1:
                if self.core_meta.is_interleaved:
                    im.shape = (ywidth, xwidth, nrgb)
                else:
                    im.shape = (nrgb, ywidth, xwidth)
                    im = np.transpose(im, (1, 2, 0))
            else:
                im.shape = (ywidth, xwidth)

            if not was_open:
                self.close()

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
        return getattr(Version, "VERSION", "unknown")
