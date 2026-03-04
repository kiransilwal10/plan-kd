"""Optional downloader for ZInD dataset placeholder."""

from __future__ import annotations

import argparse
import pathlib


def download_stub(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.txt").write_text(
        "Placeholder for ZInD dataset. Implement authenticated download if needed.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ZInD dataset (optional).")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/zind"))
    args = parser.parse_args()
    download_stub(args.out)
    print(f"ZInD stub created at {args.out}")


if __name__ == "__main__":
    main()
