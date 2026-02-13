import shutil
import sys
from pathlib import Path

import yaozarrs

from bffile import BioFile

DO_ALL = True if len(sys.argv) >= 2 and sys.argv[1] == "1" else False

for _test_file in Path("tests/data").glob("*"):
    shutil.rmtree("example.ome.zarr", ignore_errors=True)
    print("testing", _test_file)
    try:
        with BioFile(_test_file) as biofile:
            store = biofile.as_zarr_group()
            store.save("example.ome.zarr")
        yaozarrs.validate_zarr_store("example.ome.zarr")
        print("✅ Validation successful")
    except NotImplementedError as e:
        print(f"⚠️  Skipped (not supported): {e}")
    except Exception as e:
        print(f"❌ Validation failed for {_test_file}: {e}")
        if not DO_ALL:
            break
