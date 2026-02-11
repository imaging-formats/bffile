# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.32.3",
# ]
# ///
"""Fetch test data from R2 bucket."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PUBLIC_URL = "https://pub-2ac5a4b342a7472da9d44847263e1a56.r2.dev"
TEST_DATA = Path(__file__).parent.parent / "tests" / "data"
MANIFEST = "manifest.json"


def fetch_from_r2() -> list[Path]:
    """Download test data from R2 bucket. Only downloads missing files.

    Return a list of downloaded files.
    """
    TEST_DATA.mkdir(exist_ok=True, parents=True)
    manifest = requests.get(f"{PUBLIC_URL}/{MANIFEST}").json()
    local = {
        f.relative_to(TEST_DATA).as_posix() for f in TEST_DATA.rglob("*") if f.is_file()
    }

    # Download missing files
    if to_download := [f for f in manifest if f not in local]:

        def download_file(fname: str) -> str:
            content = requests.get(f"{PUBLIC_URL}/{fname}").content
            (TEST_DATA / fname).write_bytes(content)
            return fname

        with ThreadPoolExecutor(max_workers=10) as pool:
            for f in as_completed(pool.submit(download_file, f) for f in to_download):
                print(f"  -> {f.result()}")
    else:
        print("Already up to date.")

    return sorted([x for x in TEST_DATA.iterdir() if x.name != MANIFEST])


if __name__ == "__main__":
    fetch_from_r2()
