# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "boto3>=1.42.46",
# ]
# ///
"""Sync tests/data/ to R2 bucket and upload a manifest."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subprocess import check_output

import boto3

BUCKET = "bioformats-test-data"
ENDPOINT = "https://09ca0424cb878b7f78db1f0c9f9031a4.r2.cloudflarestorage.com"
OP_R2_ACCESS_KEY = "op://Private/CLOUDFLARE R2  bffile-sync/Access Key ID"
OP_R2_SECRET_KEY = "op://Private/CLOUDFLARE R2  bffile-sync/Secret Access Key"
LOCAL_DIR = Path(__file__).parent.parent / "tests" / "data"
if not LOCAL_DIR.exists():
    raise FileNotFoundError(f"Local directory does not exist: {LOCAL_DIR}")


def get_cred(env: str, op_path: str) -> str:
    cred = os.getenv(env) or check_output(["op", "read", op_path], text=True).strip()
    if not cred:
        raise ValueError(f"Missing credential: {env} / {op_path}")
    return cred


def sync() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=get_cred("R2_ACCESS_KEY", OP_R2_ACCESS_KEY),
        aws_secret_access_key=get_cred("R2_SECRET_KEY", OP_R2_SECRET_KEY),
        region_name="auto",
    )

    # Collect local files
    local = {
        f.relative_to(LOCAL_DIR).as_posix(): f
        for f in LOCAL_DIR.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    }

    # List remote files
    remote = {
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET)
        for obj in page.get("Contents", [])
    }

    # Upload missing files
    to_upload = {k: v for k, v in local.items() if k not in remote}
    if to_upload:
        with ThreadPoolExecutor(max_workers=10) as pool:
            for f in as_completed(
                pool.submit(s3.upload_file, str(p), BUCKET, r)
                for r, p in to_upload.items()
            ):
                f.result()
        print(f"Uploaded {len(to_upload)} files")

    # Delete remote files not in local
    for key in remote - local.keys() - {"manifest.json"}:
        s3.delete_object(Bucket=BUCKET, Key=key)
        print(f"Deleted {key}")

    # Upload manifest
    s3.put_object(
        Bucket=BUCKET,
        Key="manifest.json",
        Body=json.dumps(sorted(local.keys())).encode(),
    )
    print(f"✓ {len(local)} files")


if __name__ == "__main__":
    sync()
