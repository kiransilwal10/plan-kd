"""Critic filter to prune low-quality teacher labels."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Any


def is_valid(record: Dict[str, Any]) -> bool:
    evidence = record.get("evidence", {})
    rationale = record.get("notes", "") or ""
    must_ground = record.get("must_ground", False)
    tool_traces = record.get("tool_traces", {})

    if must_ground and not (evidence.get("door_ids") or evidence.get("room_ids") or evidence.get("mask") is not None):
        return False
    if rationale.lower().find("unseen") >= 0:
        return False
    if tool_traces:
        if tool_traces.get("path_len") is not None and tool_traces.get("path_len") < 0:
            return False
        if tool_traces.get("width_ok") is False:
            return False
    answer = str(record.get("answer", "")).strip().lower()
    if answer in {"", "none"}:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter teacher labels via critic rules.")
    parser.add_argument("--in", dest="inp", type=pathlib.Path, required=True)
    parser.add_argument("--out", dest="out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    total = 0
    with args.inp.open() as fin, args.out.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if is_valid(record):
                fout.write(json.dumps(record) + "\n")
                passed += 1
    print(f"Critic kept {passed}/{total} records -> {args.out}")


if __name__ == "__main__":
    main()
