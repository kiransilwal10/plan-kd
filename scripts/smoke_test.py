"""Smoke test: runs 1 epoch on 4 synthetic samples through all three training stages.

Verifies each stage runs without crashing and produces its checkpoint.
No GPU or real images required — runs entirely on CPU with a temp directory.

Usage (from repo root):
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).parent.parent  # plan-kd/

# ── Synthetic JSONL samples ───────────────────────────────────────────────────
# image_path is intentionally missing — load_pixels() falls back to a white image.
SAMPLES = [
    {
        "qid": f"q{i}",
        "image_id": f"img{i}",
        "image_path": "/nonexistent/fake.png",
        "question_key": "door_count",
        "question": "How many doors are in this floor plan?",
        "answer": "two",
        "evidence": {"door_ids": ["d1", "d2"], "room_ids": ["r1"]},
        "uncertainty": 0.1,
        "must_ground": True,
        "notes": "",
    }
    for i in range(4)
]

# ── Config templates ──────────────────────────────────────────────────────────
DATA_CFG_TMPL = """\
image_root: data/waffle
image_ext: [.png, .jpg]
splits:
  train: {train_jsonl}
  val: {train_jsonl}
  test: {train_jsonl}
"""

TRAIN_A_CFG_TMPL = """\
seed: 42
data_cfg: {data_cfg}
model_cfg: configs/model.yaml
output_root: {output_root}
stage: answer_kd
trainer:
  epochs: 1
  batch_size: 2
  grad_accum: 1
  num_workers: 0
  log_interval: 1
  save_every: 9999
  device: cpu
loss:
  ce_weight: 1.0
  kd_weight: 1.0
  temperature: 2.0
"""

TRAIN_B_CFG_TMPL = """\
seed: 42
data_cfg: {data_cfg}
model_cfg: configs/model.yaml
output_root: {output_root}
stage: evidence_kd
trainer:
  epochs: 1
  batch_size: 2
  grad_accum: 1
  num_workers: 0
  log_interval: 1
  save_every: 9999
  device: cpu
loss:
  ce_weight: 1.0
  kd_weight: 1.0
  evidence_weight: 1.0
  temperature: 2.0
"""

TRAIN_C_CFG_TMPL = """\
seed: 42
data_cfg: {data_cfg}
model_cfg: configs/model.yaml
output_root: {output_root}
stage: grpo_unknown
trainer:
  epochs: 1
  batch_size: 2
  grad_accum: 1
  num_workers: 0
  log_interval: 1
  save_every: 9999
  device: cpu
grpo:
  group_size: 2
  lr: 1.0e-5
  clip_eps: 0.2
  kl_weight: 0.01
  reward:
    answer_weight: 0.5
    evidence_weight: 0.3
    abstain_weight: 0.2
"""


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[smoke] FAILED at: {cmd[2]}")
        sys.exit(result.returncode)


def main() -> None:
    print("[smoke] Starting smoke test — CPU only, synthetic data, temp dir.")

    with tempfile.TemporaryDirectory(prefix="plan_kd_smoke_") as _tmp:
        tmp = pathlib.Path(_tmp)

        # Write synthetic JSONL
        train_jsonl = tmp / "train.jsonl"
        train_jsonl.write_text("\n".join(json.dumps(s) for s in SAMPLES))

        # Write data config
        data_cfg_path = tmp / "data.yaml"
        data_cfg_path.write_text(DATA_CFG_TMPL.format(train_jsonl=train_jsonl))

        out_root = tmp / "outputs"

        # ── Stage A ──────────────────────────────────────────────────────────
        print("\n[smoke] Stage A — answer distillation")
        cfg_a = tmp / "trainA.yaml"
        cfg_a.write_text(TRAIN_A_CFG_TMPL.format(data_cfg=data_cfg_path, output_root=out_root))
        run([sys.executable, "-m", "student.train_stageA_answer_kd", "--cfg", cfg_a])

        ckpt_a = sorted(out_root.glob("*/student-A.pt"))
        if not ckpt_a:
            print("[smoke] FAILED: student-A.pt not found.")
            sys.exit(1)
        ckpt_a = ckpt_a[-1]
        print(f"[smoke] PASS Stage A  →  {ckpt_a.name}")

        # ── Stage B ──────────────────────────────────────────────────────────
        print("\n[smoke] Stage B — evidence distillation")
        cfg_b = tmp / "trainB.yaml"
        cfg_b.write_text(TRAIN_B_CFG_TMPL.format(data_cfg=data_cfg_path, output_root=out_root))
        run([sys.executable, "-m", "student.train_stageB_evidence_kd", "--cfg", cfg_b, "--ckpt", ckpt_a])

        ckpt_b = sorted(out_root.glob("*/student-B.pt"))
        if not ckpt_b:
            print("[smoke] FAILED: student-B.pt not found.")
            sys.exit(1)
        ckpt_b = ckpt_b[-1]
        print(f"[smoke] PASS Stage B  →  {ckpt_b.name}")

        # ── Stage C ──────────────────────────────────────────────────────────
        print("\n[smoke] Stage C — GRPO")
        cfg_c = tmp / "trainC.yaml"
        cfg_c.write_text(TRAIN_C_CFG_TMPL.format(data_cfg=data_cfg_path, output_root=out_root))
        run([sys.executable, "-m", "student.train_stageC_dpo_unknown", "--cfg", cfg_c, "--ckpt", ckpt_b])

        ckpt_c = sorted(out_root.glob("*/student-C.pt"))
        if not ckpt_c:
            print("[smoke] FAILED: student-C.pt not found.")
            sys.exit(1)
        ckpt_c = ckpt_c[-1]
        print(f"[smoke] PASS Stage C  →  {ckpt_c.name}")

        print("\n[smoke] All 3 stages passed.")
    # temp dir cleaned up automatically


if __name__ == "__main__":
    main()
