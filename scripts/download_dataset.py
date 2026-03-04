"""Download public datasets with integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request
import zipfile
from typing import Tuple


DATASETS = {
    "cubicasa5k": {
        "meta_url": "https://zenodo.org/api/records/2613548",
        "download_url": "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content",
        "filename": "cubicasa5k.zip",
        "extract_to": pathlib.Path("data/waffle"),
    },
}


def fetch_checksum(meta_url: str, filename: str) -> Tuple[str, str]:
    with urllib.request.urlopen(meta_url) as resp:
        meta = json.load(resp)
    for file_info in meta.get("files", []):
        if file_info.get("key", "").lower() == filename.lower():
            checksum = file_info.get("checksum", "")
            if ":" in checksum:
                alg, val = checksum.split(":", 1)
                return alg.lower(), val
    raise RuntimeError(f"Checksum for {filename} not found at {meta_url}")


def download_file(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)


def compute_checksum(path: pathlib.Path, alg: str) -> str:
    h = hashlib.new(alg)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(zip_path: pathlib.Path, target_dir: pathlib.Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset with integrity check.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--force", action="store_true", help="Re-download even if zip exists.")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    zip_path = pathlib.Path("data") / cfg["filename"]

    if zip_path.exists() and not args.force:
        print(f"{zip_path} already exists; use --force to re-download.")
    else:
        print(f"Fetching checksum from {cfg['meta_url']}...")
        alg, expected = fetch_checksum(cfg["meta_url"], cfg["filename"])
        print(f"Downloading {cfg['download_url']} -> {zip_path}")
        download_file(cfg["download_url"], zip_path)
        actual = compute_checksum(zip_path, alg)
        if actual != expected:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {zip_path}: expected {expected}, got {actual}")
        print(f"Checksum OK ({alg}={actual})")

    print(f"Extracting {zip_path} to {cfg['extract_to']}")
    extract_zip(zip_path, cfg["extract_to"])
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
