from __future__ import annotations

import os
import warnings
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, cast

import dask.array as da
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
        self._path = str(Path(path).expanduser().absolute())
        self._current_scene_index = series
        self._lock = Lock()
        self.dask_tiles = dask_tiles

        self._java_reader = jimport("loci.formats.ImageReader")()
        if meta:
            self._java_reader.setMetadataStore(self._create_ome_meta())
        if original_meta:
            self._java_reader.setOriginalMetadataPopulated(True)

        # memoize to save time on later re-openings of the same file.
        if memoize > 0:
            Memoizer = jimport("loci.formats.Memoizer")
            if BIOFORMATS_MEMO_DIR is not None:
                self._java_reader = Memoizer(
                    self._java_reader, memoize, BIOFORMATS_MEMO_DIR
                )
            else:
                self._java_reader = Memoizer(self._java_reader, memoize)

        if options:
            DynamicMetadataOptions = jimport("loci.formats.in_.DynamicMetadataOptions")
            mo = DynamicMetadataOptions()
            for name, value in options.items():
                mo.set(name, str(value))
            self._java_reader.setMetadataOptions(mo)

        self.open()
        self.set_series(series)

        if self.dask_tiles:
            if tile_size is None:
                self.tile_size = (
                    self._java_reader.getOptimalTileHeight(),
                    self._java_reader.getOptimalTileWidth(),
                )
            else:
                if len(tiles := tuple(tile_size)) != 2:
                    raise ValueError(f"tile_size must be length 2, got {len(tiles)}")
                if not all(isinstance(x, int) for x in tiles):
                    raise ValueError(f"tile_size must be integers, got {tiles}")
                self.tile_size = cast("tuple[int, int]", tiles)

    def set_series(self, series: int = 0) -> None:
        self._java_reader.setSeries(series)
        self._current_scene_index = series

    def java_reader(self) -> IFormatReader:
        """Return the native reader object."""
        return self._java_reader

    @property
    def core_meta(self) -> CoreMetadata:
        return self._core_meta_list[self._current_scene_index]

    @property
    def shape(self) -> OMEShape:
        return self.core_meta.shape

    @property
    def dtype(self) -> np.dtype:
        return self.core_meta.dtype

    def open(self) -> None:
        """Open file."""
        self._java_reader.setId(self._path)
        self._java_reader.setSeries(self._current_scene_index)
        self._core_meta_list = [
            CoreMetadata.from_java(x) for x in self._java_reader.getCoreMetadataList()
        ]

    def close(self) -> None:
        """Close file."""
        with suppress(Exception):
            self._java_reader.close()

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
        """Whether the underlying file is currently open."""
        return not bool(self._java_reader.getCurrentFile())

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        # return self._r.getCurrentFile()
        return self._path

    @property
    def ome_xml(self) -> str:
        """Return OME XML string."""
        if store := self._java_reader.getMetadataStore():
            try:
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
        return str(getattr(Version, "VERSION", "unknown"))
