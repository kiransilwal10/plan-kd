"""Generate teacher Q/A + evidence labels via OpenAI vision models.

Usage:
    python teacher/generate_labels.py --cfg configs/data.yaml --out data/labels/train.jsonl

Requirements:
    - Set OPENAI_API_KEY in your environment.
    - The dataset (e.g., CubiCasa5k) should be downloaded under data/waffle.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm
from PIL import Image

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


QUESTION_BANK: List[Tuple[str, str]] = [
    ("door_count", "How many doors are visible in the floor plan?"),
    ("entry", "Is there a clear main entrance door visible in the plan? Answer yes or no."),
    ("egress", "Is there an unobstructed exit path from the largest room to the outside? Answer yes or no."),
    ("room_count", "How many distinct rooms are visible?"),
    ("kitchen_loc", "Where is the kitchen located relative to the main entrance? Keep it short."),
    ("bath_count", "How many bathrooms can you see?"),
    ("stairs", "Do you see any stairs? Answer yes or no."),
    ("elevator", "Is there an elevator drawn anywhere? Answer yes or no."),
    ("garage", "Is there a garage or car space shown? Answer yes or no."),
    ("balcony", "Do you see a balcony or terrace? Answer yes or no."),
    ("bed_count", "How many bedrooms are labeled or clearly drawn?"),
    ("hall_width", "Is the narrowest hallway wide enough for one person to pass comfortably? Answer yes, no, or unknown."),
    ("window_check", "Is there any room without a window? Answer yes, no, or unknown."),
    ("open_plan", "Is the living+dining area an open plan (no separating walls)? Answer yes or no."),
    ("laundry", "Is there a laundry or washer/dryer area marked? Answer yes or no."),
    ("odd_question", "Point to the weirdest-looking room or space on this plan and say what makes it weird."),
    ("bath_adjacent", "Is any bathroom directly connected to a bedroom? Answer yes or no."),
    ("hall_loop", "Does the hallway form a loop that lets you walk a circle? Answer yes or no."),
    ("longest_room", "Which room looks longest side-to-side? Name it briefly."),
    ("largest_room", "Which room appears largest by area?"),
    ("smallest_room", "Which room appears smallest by area?"),
    ("corner_room", "Which room sits in the top-left corner of the plan?"),
    ("outdoor_access", "Is there direct outdoor access from a living space (not through a bedroom)? Answer yes or no."),
    ("double_doors", "Do you see any double doors? Answer yes or no."),
    ("sliding_door", "Is there a sliding door drawn anywhere? Answer yes or no."),
    ("kitchen_island", "Does the kitchen show an island? Answer yes or no."),
    ("pantry", "Is there a pantry or storage closet near the kitchen? Answer yes or no."),
    ("fireplace", "Is a fireplace indicated? Answer yes or no."),
    ("desk_space", "Is there a dedicated office/desk space? Answer yes or no."),
    ("closet_count", "Roughly how many closets are shown? Give a number."),
    ("corridor_doors", "How many doors open off the main corridor?"),
    ("entrance_view", "From the entrance, what is the first room you would step into or see?"),
    ("bath_group", "Are the bathrooms clustered together or spread out? Answer clustered, spread, or unknown."),
    ("split_level", "Does the plan imply split levels (half-stair or elevation change)? Answer yes or no."),
    ("ceiling_feature", "Do you see any special ceiling notes (vaulted/beam)? Answer yes or no."),
    ("utility", "Is there a mechanical/utility room? Answer yes or no."),
    ("storage_outdoor", "Is there any dedicated outdoor storage or shed shown? Answer yes or no."),
    ("kids_room", "Is any room labeled for kids/child/nursery? Answer yes or no."),
    ("guest_room", "Is there a guest room or multipurpose room marked? Answer yes or no."),
    ("shape", "Roughly what shape is the overall floor plan outline? (e.g., rectangle, L-shape)"),
    ("hall_turns", "How many 90-degree turns does the main hallway make? Give a number."),
    ("front_back", "Which side of the plan looks like the front (street) side?"),
    ("porch", "Is there a porch or covered entry? Answer yes or no."),
    ("deck", "Is there a deck or patio? Answer yes or no."),
    ("bath_tub", "Do any bathrooms clearly show a bathtub? Answer yes or no."),
    ("shower_only", "Is there any bathroom that appears shower-only? Answer yes or no."),
    ("ensuite", "Is there an ensuite bath attached to the primary bedroom? Answer yes or no."),
    ("primary_bed", "Which room seems to be the primary bedroom? Describe briefly."),
    ("traffic_flow", "Is traffic flow mostly linear (no backtracking) from entrance to bedrooms? Answer yes or no."),
    ("accessibility", "Do doorways/hallways appear wide enough for wheelchair access? Answer yes, no, or unknown."),
    ("duplicate_spaces", "Are there two rooms serving the same function (e.g., two living rooms)? Answer yes or no."),
    ("kitchen_sink", "Can you spot the kitchen sink placement? Answer yes or unknown."),
    ("island_seating", "If there is an island, does it show seating? Answer yes or no."),
    ("foyer", "Is there a foyer/entry hall distinct from the living space? Answer yes or no."),
    ("mudroom", "Is there a mudroom/drop zone near the entry or garage? Answer yes or no."),
    ("linen_closet", "Is there a linen closet near bedrooms/baths? Answer yes or no."),
    ("coat_closet", "Is there a coat closet near the entry? Answer yes or no."),
    ("room_labels", "Do most rooms have text labels? Answer yes or no."),
    ("furniture_sparse", "Is the furniture layout sparse (few symbols) or dense? Answer sparse, dense, or none."),
    ("out_of_place", "Is there any symbol or room that feels out of place for a home? Briefly name it."),
    ("vibes_weird", "Weird vibes? yes/no"),
    ("just_bedrooms", "5 beds?"),
    ("no_doors", "Any doorless rooms?"),
    ("too_many_doors", "Too many doors? yes/no"),
    ("family_fit", "Family of 5 ok here?"),
    ("party_house", "Party house or nah?"),
    ("pet_corner", "Pet corner?"),
    ("nook", "Cozy nook somewhere?"),
    ("big_kitchen", "Is the kitchen big?"),
    ("long_hall", "Long hallway?"),
    ("straight_line", "Can I walk straight from entry to living? yes/no"),
    ("garage_y_n", "Garage y/n"),
    ("guest_ok", "Good for guests? yes/no"),
    ("windowless", "Windowless room? yes/no"),
    ("sleep_5", "Sleep 5 comfortably?"),
    ("office_ready", "WFH ready? yes/no"),
    ("odd_two_words", "weird nook?"),
    ("bedroom_hungry", "I wanna 5 bedroom, is this for me?"),
    ("messy_question", "uh is there like a tiny room nobody uses?"),
    ("short_blurt", "hall weird?"),
    ("half_sentence", "doors too many or nah?"),
]


@dataclass
class Label:
    qid: str
    image_id: str
    image_path: str
    question_key: str
    question: str
    answer: str
    evidence: Dict[str, Any]
    notes: str
    must_ground: bool
    uncertainty: float
    tool_traces: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def load_prompt(prompt_path: pathlib.Path) -> Tuple[str, str]:
    text = prompt_path.read_text()
    system_prompt = "You are a cautious plan reviewer. Only answer if visible; otherwise output `unknown`. Provide evidence IDs and a one-line rationale."
    user_template = "Return JSON with fields: answer, evidence (door_ids, room_ids, mask), rationale."
    if "System:" in text and "User" in text:
        parts = text.split("User", 1)
        system_prompt = parts[0].replace("System:", "").strip()
        user_template = parts[1].replace("(template):", "").strip()
    return system_prompt, user_template


def encode_image(path: pathlib.Path, max_size: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def choose_questions(idx: int, k: int = 3) -> List[Tuple[str, str]]:
    rng = random.Random(idx)
    if k >= len(QUESTION_BANK):
        return QUESTION_BANK
    return rng.sample(QUESTION_BANK, k)


def make_image_id(img_path: pathlib.Path, root: pathlib.Path) -> str:
    """Return a stable, unique ID per image using path relative to dataset root."""
    try:
        rel = img_path.resolve().relative_to(root.resolve())
    except Exception:
        rel = img_path.name
    rel_str = rel.as_posix() if hasattr(rel, "as_posix") else str(rel)
    return rel_str.replace(" ", "_")


def load_existing_counts(out_path: pathlib.Path) -> Dict[str, int]:
    """Return how many questions each image_id already has in the JSONL."""
    counts: Dict[str, int] = {}
    if not out_path.exists():
        return counts
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            img_id = record.get("image_id")
            if img_id is None:
                continue
            counts[img_id] = counts.get(img_id, 0) + 1
    return counts


def call_teacher(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    image_data_url: str,
    temperature: float,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=1,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"[error] Non-JSON response from teacher: {content}", file=sys.stderr)
        raise RuntimeError(f"Non-JSON response from teacher: {content[:200]}") from exc


def fallback_label(question_key: str, question: str, img_id: str, img_path: pathlib.Path, q_idx: int) -> Label:
    return Label(
        qid=f"{img_id}-q{q_idx}",
        image_id=img_id,
        image_path=str(img_path),
        question_key=question_key,
        question=question,
        answer="unknown",
        evidence={"door_ids": [], "room_ids": [], "mask": None},
        notes="Fallback label (no API).",
        must_ground=True,
        uncertainty=1.0,
        tool_traces={"path_len": None, "width_ok": None},
    )


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def build_label(record: Dict[str, Any], question_key: str, question: str, img_id: str, img_path: pathlib.Path, q_idx: int) -> Label:
    evidence = record.get("evidence", {}) if isinstance(record, dict) else {}
    notes = record.get("notes", "") if isinstance(record, dict) else ""
    must_ground = bool(record.get("must_ground", True)) if isinstance(record, dict) else True
    uncertainty = safe_float(record.get("uncertainty", 0.0), default=0.0) if isinstance(record, dict) else 0.0
    tool_traces = record.get("tool_traces", {}) if isinstance(record, dict) else {}
    answer = str(record.get("answer", "unknown")) if isinstance(record, dict) else "unknown"
    return Label(
        qid=f"{img_id}-q{q_idx}",
        image_id=img_id,
        image_path=str(img_path),
        question_key=question_key,
        question=question,
        answer=answer,
        evidence={
            "door_ids": evidence.get("door_ids", []),
            "room_ids": evidence.get("room_ids", []),
            "mask": evidence.get("mask"),
        },
        notes=notes or "Teacher response",
        must_ground=must_ground,
        uncertainty=uncertainty,
        tool_traces=tool_traces or {"path_len": None, "width_ok": None},
    )


def gather_image_paths(data_cfg: Dict[str, Any]) -> List[pathlib.Path]:
    raster_path = pathlib.Path(data_cfg["datasets"]["raster"]["path"])
    exts = data_cfg["datasets"]["raster"].get("extensions", [".jpg", ".png", ".jpeg"])
    paths: List[pathlib.Path] = []
    for ext in exts:
        paths.extend(sorted(raster_path.rglob(f"*{ext}")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher labels with OpenAI VLM.")
    parser.add_argument("--cfg", type=pathlib.Path, required=True, help="Data config (for locating images).")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Output JSONL path.")
    parser.add_argument("--prompt", type=pathlib.Path, default=pathlib.Path("teacher/prompts/general.txt"))
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_completion_tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=200, help="Number of images to process.")
    parser.add_argument("--per_image", type=int, default=3, help="Questions per image.")
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.cfg.read_text())
    system_prompt, user_template = load_prompt(args.prompt)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY must be set and openai package available for teacher generation.")
    client = OpenAI(api_key=api_key)

    raster_root = pathlib.Path(data_cfg["datasets"]["raster"]["path"]).resolve()
    image_paths = [p.resolve() for p in gather_image_paths(data_cfg)]
    if not image_paths:
        print("No images found. Ensure data/waffle contains raster plans.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing_counts = load_existing_counts(args.out)
    write_mode = "a" if args.out.exists() else "w"

    to_process = image_paths[: args.limit]
    skip_full = 0
    for img_path in to_process:
        img_id = make_image_id(img_path, raster_root)
        if existing_counts.get(img_id, 0) >= args.per_image:
            skip_full += 1
    print(
        f"Found {len(image_paths)} images (limit {args.limit}); "
        f"{skip_full} already have >= {args.per_image} labels; "
        f"processing up to {len(to_process) - skip_full} images."
    )

    generated = 0
    skipped = 0

    with args.out.open(write_mode) as f:
        for img_idx, img_path in enumerate(tqdm(to_process, desc="Images")):
            img_id = make_image_id(img_path, raster_root)
            questions = choose_questions(img_idx, k=args.per_image)
            have = existing_counts.get(img_id, 0)
            if have >= args.per_image:
                skipped += 1
                continue
            for q_local_idx, (q_key, question) in enumerate(questions):
                if q_local_idx < have:
                    continue
                if client is None:
                    label = fallback_label(q_key, question, img_id, img_path, q_local_idx)
                    f.write(label.to_json() + "\n")
                    generated += 1
                    continue

                try:
                    img_b64 = encode_image(img_path, max_size=data_cfg["datasets"]["raster"].get("image_size", 512))
                    user_text = (
                        f"Task type: {q_key}\n"
                        f"Question: {question}\n"
                        "If unsure, answer 'unknown'. Follow the JSON schema: "
                        "{answer, evidence:{door_ids, room_ids, mask}, notes, must_ground, uncertainty, tool_traces}."
                    )
                    record = call_teacher(
                        client=client,
                        model=args.model,
                        system_prompt=system_prompt,
                        user_text=user_text,
                        image_data_url=img_b64,
                        temperature=args.temperature,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                    label = build_label(record, q_key, question, img_id, img_path, q_local_idx)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] {img_path} ({q_key}): {exc}", file=sys.stderr)
                    label = fallback_label(q_key, question, img_id, img_path, q_local_idx)
                f.write(label.to_json() + "\n")
                generated += 1

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"Completed generation to {args.out} [{date_tag}] using model {args.model}. "
        f"Generated {generated} entries; skipped {skipped} images already filled."
    )


if __name__ == "__main__":
    main()
