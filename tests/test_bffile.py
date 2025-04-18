from pathlib import Path

from bioformats import BioFile


def test_bioformats(test_file: Path) -> None:
    with BioFile(test_file) as bf:
        assert bf.to_numpy() is not None
        assert bf.to_dask() is not None
        assert bf.shape
