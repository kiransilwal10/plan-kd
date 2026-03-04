#!/usr/bin/env bash
# Full three-stage training pipeline: A (answer KD) → B (evidence KD) → C (GRPO)
# Usage: bash train_all.sh [--stageA-only] [--from-ckpt-A <path>] [--from-ckpt-B <path>]

set -euo pipefail

STAGE_A_ONLY=false
CKPT_A=""
CKPT_B=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stageA-only)   STAGE_A_ONLY=true; shift ;;
        --from-ckpt-A)   CKPT_A="$2"; shift 2 ;;
        --from-ckpt-B)   CKPT_B="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Stage A ────────────────────────────────────────────────────────────────
if [[ -z "$CKPT_A" && -z "$CKPT_B" ]]; then
    echo "==> Stage A: answer distillation"
    python -m student.train_stageA_answer_kd --cfg configs/trainA.yaml

    CKPT_A=$(ls -t outputs/*/student-A.pt 2>/dev/null | head -1)
    if [[ -z "$CKPT_A" ]]; then
        echo "ERROR: Stage A checkpoint not found under outputs/"; exit 1
    fi
    echo "    checkpoint: $CKPT_A"
fi

[[ "$STAGE_A_ONLY" == true ]] && echo "Done (Stage A only)." && exit 0

# ── Stage B ────────────────────────────────────────────────────────────────
if [[ -z "$CKPT_B" ]]; then
    echo "==> Stage B: evidence distillation"
    python -m student.train_stageB_evidence_kd --cfg configs/trainB.yaml --ckpt "$CKPT_A"

    CKPT_B=$(ls -t outputs/*/student-B.pt 2>/dev/null | head -1)
    if [[ -z "$CKPT_B" ]]; then
        echo "ERROR: Stage B checkpoint not found under outputs/"; exit 1
    fi
    echo "    checkpoint: $CKPT_B"
fi

# ── Stage C ────────────────────────────────────────────────────────────────
echo "==> Stage C: GRPO"
python -m student.train_stageC_dpo_unknown --cfg configs/trainC.yaml --ckpt "$CKPT_B"

CKPT_C=$(ls -t outputs/*/student-C.pt 2>/dev/null | head -1)
if [[ -z "$CKPT_C" ]]; then
    echo "ERROR: Stage C checkpoint not found under outputs/"; exit 1
fi
echo "    checkpoint: $CKPT_C"

echo "==> Training complete. Final model: $CKPT_C"
