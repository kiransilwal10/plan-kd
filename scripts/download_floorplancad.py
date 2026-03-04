"""Stub downloader for FloorPlanCAD (SVG + JSON)."""

from __future__ import annotations

import argparse
import pathlib


def download_stub(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.txt").write_text(
        "Placeholder for FloorPlanCAD dataset.\n"
        "Add SVG and JSON files here once downloaded.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FloorPlanCAD dataset.")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/floorplancad"))
    args = parser.parse_args()
    download_stub(args.out)
    print(f"FloorPlanCAD stub created at {args.out}")


if __name__ == "__main__":
    main()
