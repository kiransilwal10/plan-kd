"""Stub downloader for WAFFLE raster dataset.

Replace the placeholder download logic with actual dataset retrieval.
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.request


def download_stub(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "README.txt"
    marker.write_text(
        "Placeholder for WAFFLE raster dataset.\n"
        "Replace this file after implementing real download logic.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WAFFLE raster dataset.")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/waffle"))
    args = parser.parse_args()
    download_stub(args.out)
    print(f"WAFFLE stub assets placed under {args.out}")


if __name__ == "__main__":
    main()
