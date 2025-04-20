from pathlib import Path

import ome_types
import pytest

from bffile import BioFile


@pytest.mark.parametrize('memoize', [True, False])
def test_bffile(test_file: Path, memoize: bool) -> None:
    with BioFile(test_file, memoize=False) as bf:
        assert bf.to_numpy() is not None
        assert bf.to_dask() is not None
        assert isinstance(bf.ome_metadata, ome_types.OME)
        assert bf.shape
