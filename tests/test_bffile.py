from pathlib import Path

import ome_types

from bffile import BioFile


def test_bffile(test_file: Path) -> None:
    with BioFile(test_file, original_meta=True) as bf:
        assert bf.to_numpy() is not None
        assert bf.to_dask() is not None
        assert isinstance(bf.ome_metadata, ome_types.OME)
        assert bf.shape
