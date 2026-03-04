"""Parse vector CAD plans (SVG + JSON metadata) into unified tokens."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Any

from svgpathtools import svg2paths2


def svg_to_tokens(svg_path: pathlib.Path) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    paths, attributes, svg_attr = svg2paths2(str(svg_path))
    for idx, (path, attr) in enumerate(zip(paths, attributes)):
        tokens.append(
            {
                "id": f"path-{idx}",
                "type": attr.get("id", "path"),
                "geom": attr.get("d", ""),
                "attrs": attr,
            }
        )
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse vector CAD floor-plans.")
    parser.add_argument("--data_root", type=pathlib.Path, default=pathlib.Path("data/floorplancad"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/cache/vector_index.json"))
    parser.add_argument("--json_ext", type=str, default=".json")
    parser.add_argument("--svg_ext", type=str, default=".svg")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for svg_path in sorted(args.data_root.glob(f"*{args.svg_ext}")):
        sample = {
            "source": str(svg_path),
            "cad_tokens": svg_to_tokens(svg_path),
            "room_polys": [],
            "door_polys": [],
            "ocr_text": [],
            "meta": {"rotation": 0},
        }
        json_path = svg_path.with_suffix(args.json_ext)
        if json_path.exists():
            with json_path.open() as jf:
                try:
                    sample["meta"].update(json.load(jf))
                except json.JSONDecodeError:
                    pass
        records.append(sample)

    args.out.write_text(json.dumps({"count": len(records), "samples": records}, indent=2))
    print(f"Parsed {len(records)} vector plans -> {args.out}")


if __name__ == "__main__":
    main()
