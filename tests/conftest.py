import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import jpype
import pytest
import scyjava

from bffile import _biofile

TEST_DATA = Path(__file__).parent / "data"
if not TEST_DATA.exists() or not any(TEST_DATA.iterdir()):
    print("Test data not found, fetching from R2...")
    script = Path(__file__).parent.parent / "scripts" / "fetch_test_data.py"
    subprocess.run([sys.executable, str(script)], check=True)
DATA_FILES = sorted(TEST_DATA.iterdir())


@pytest.fixture(scope="session", params=DATA_FILES, ids=lambda x: x.name)
def any_file(request: pytest.FixtureRequest) -> Path:
    """Fixture to provide a test file from the data folder."""
    return request.param


@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    return TEST_DATA


@pytest.fixture
def simple_file() -> Path:
    """Small TIFF file for fast unit tests (7KB)."""
    return TEST_DATA / "s_1_t_1_c_1_z_1.ome.tiff"


@pytest.fixture
def multiseries_file() -> Path:
    """ND2 file with 4 series for multi-series testing."""
    return TEST_DATA / "ND2_dims_p4z5t3c2y32x32.nd2"


@pytest.fixture
def pyramid_file() -> Path:
    """SVS file with multiple resolution levels."""
    return TEST_DATA / "CMU-1.svs"


@pytest.fixture
def rgb_file() -> Path:
    """RGB file for testing."""
    return TEST_DATA / "RGB-8bit.czi"


@pytest.fixture(
    params=[
        "s_1_t_1_c_2_z_1_RGB.tiff",
        "ND2_dims_c2y32x32.nd2",
        "s_1_t_1_c_1_z_1.ome.tiff",
    ]
)
def small_test_file(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture with small test files for format coverage."""
    return TEST_DATA / request.param


@pytest.fixture
def opened_biofile(
    request: pytest.FixtureRequest, simple_file: Path
) -> Iterator[_biofile.BioFile]:
    """Pre-opened BioFile instance for convenience in tests.

    Uses simple_file by default, or parametrized with all files via --exhaustive.
    """
    # If parametrized (exhaustive mode), use the parameter
    if hasattr(request, "param"):
        file_path = request.param
    else:
        file_path = simple_file

    bf = _biofile.BioFile(file_path)
    bf.open()
    yield bf
    bf.close()


def pytest_sessionstart() -> None:
    # https://jpype.readthedocs.io/en/latest/userguide.html#errors-reported-by-python-fault-handler
    import faulthandler

    faulthandler.enable()
    faulthandler.disable()


# register pytest options
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-jgo-cache",
        action="store_true",
        default=False,
        help="Allow cache to be used for tests",
    )
    parser.addoption(
        "--test-java-constraints",
        action="store_true",
        default=False,
        help="Run slow integration tests for Java constraints",
    )
    parser.addoption(
        "--exhaustive",
        action="store_true",
        default=False,
        help="Run exhaustive tests using all test files (slow)",
    )


@pytest.fixture(autouse=True, scope="session")
def memo_dir(request: pytest.FixtureRequest) -> Iterator[Path | None]:
    """Ensure we test on a clean cache, and don't pollute the user's cache."""
    tmp_path = Path(tempfile.mkdtemp())
    atexit.register(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    _biofile.BIOFORMATS_MEMO_DIR = tmp_path / "memo"
    yield tmp_path


@pytest.fixture(autouse=True, scope="session")
def cache_dirs(request: pytest.FixtureRequest) -> Iterator[Path | None]:
    """Ensure we test on a clean cache, and don't pollute the user's cache."""

    # caching significantly speeds up tests, but we want to ensure that
    # we test on a clean cache, and don't pollute the user's cache.
    if not request.config.getoption("--no-jgo-cache"):
        print("Reading/writing to actual user cache directories")
        yield None
        return

    tmp_path = Path(tempfile.mkdtemp())
    os.environ["CJDK_CACHE_DIR"] = str(tmp_path / "cjdk")
    os.environ["JGO_CACHE_DIR"] = jgo = str(tmp_path / "jgo")
    scyjava.config.set_cache_dir(jgo)

    m2_repo = tmp_path / "m2" / "repository"
    m2_repo.mkdir(parents=True, exist_ok=True)
    os.environ["M2_REPO"] = _m2_repo = str(m2_repo)
    os.environ["MAVEN_OPTS"] = f"-Dmaven.repo.local={_m2_repo}"
    scyjava.config.set_m2_repo(_m2_repo)

    # doing manual cleanup here to avoid windows file locks on jars during teardown
    @atexit.register
    def _cleanup() -> None:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
        shutil.rmtree(tmp_path, ignore_errors=True)

    yield tmp_path


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize opened_biofile based on --exhaustive flag."""
    if "opened_biofile" in metafunc.fixturenames:
        if metafunc.config.getoption("--exhaustive"):
            # Parametrize with all data files
            metafunc.parametrize(
                "opened_biofile",
                DATA_FILES,
                indirect=True,
                ids=lambda x: x.name,
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Iterator[None]:
    """Convert certain exceptions to xfail/skip."""
    outcome = yield
    report = outcome.get_result()

    # Convert UnknownFormatException failures to xfail
    if report.when == "call" and report.failed:
        if call.excinfo is not None:
            exc_value = str(call.excinfo.value)
            if "UnknownFormatException" in exc_value:
                report.outcome = "skipped"
                data = Path(__file__).parent / "data"
                report.wasxfail = str(exc_value).replace(str(data), "")
