"""Package outputs and configs for release."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import tarfile
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Package outputs and configs.")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs"))
    args = parser.parse_args()

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = pathlib.Path(f"outputs/{date_tag}")
    bundle_root.mkdir(parents=True, exist_ok=True)
    archive_path = bundle_root / "release.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        for rel in ["configs", "Makefile", "README.md", "requirements.txt", "teacher", "student", "eval", "tools"]:
            path = pathlib.Path(rel)
            if path.exists():
                tar.add(path, arcname=path)
        outputs_dir = pathlib.Path("outputs")
        for child in outputs_dir.iterdir():
            if child.is_dir() and child.name != date_tag:
                tar.add(child, arcname=f"outputs/{child.name}")
    print(f"Release bundle created at {archive_path}")


if __name__ == "__main__":
    main()
