# Floor-Plan VLM Distillation: Method and Design Choices

This document summarizes the end-to-end pipeline to distill a large VLM into a tiny model for evidence-grounded floor-plan QA. It covers data, teacher, student, losses, training stages, evaluation, and key design decisions.

## Goal
Train a compact VLM (few billion parameters or smaller) to answer floor-plan questions with evidence pointers and abstention, using distilled labels from a stronger teacher model.

## Data & Representation
- **Inputs**: Floor-plan images (raster PNG/JPG) and optional vector CAD tokens.
- **Sample schema**: `plan_image` (Tensor), `cad_tokens`, `room_polys`, `door_polys`, `meta` (includes `image_path`, rotation, etc.).
- **Label schema (JSONL)**: `{ qid, image_id, image_path, question_key, question, answer, evidence:{door_ids, room_ids, mask}, notes, must_ground, uncertainty, tool_traces }`.
- **Splits**: `data/labels/train.filtered.jsonl`, `val.filtered.jsonl`, `test.filtered.jsonl`.

## Teacher Generation
- **Model**: OpenAI vision chat model (default `gpt-4o-mini`), templated prompt in `teacher/prompts/general.txt`.
- **Questions**: A large bank of diverse/quirky prompts; 3 sampled per image (deterministic by image index).
- **Evidence expectation**: Answer + evidence IDs (doors/rooms) + rationale; abstain with `unknown` if unsure.
- **Execution**: `make teacher-all` (config-driven, uses `OPENAI_API_KEY`). Skips images already labeled (based on `image_id`).
- **Critic filter**: `make labels` prunes inconsistent/missing-evidence samples.
- **Progress logging**: tqdm over images, and summary of generated/skipped entries.

## Student Model (Current)
- **Vision backbone**: CLIP vision (`openai/clip-vit-base-patch32` by default), optionally frozen.
- **Text backbone**: Small pretrained LM (default `gpt2`), optionally partially frozen/LoRA-able.
- **Fusion**: Concatenate pooled vision + pooled text; project via MLP to shared hidden; heads for answer logits, evidence pointer logits, abstain logit.
- **Tokenizer**: Uses the text model tokenizer; replaces earlier char-level vocab.
- **Image preproc**: CLIPImageProcessor; images loaded from `image_path` in labels.

## Losses
- **Answer KD (Stage A)**: Cross-entropy to target token + KL to teacher logits (temperature-scaled).
- **Evidence KD (Stage B)**: Adds MSE/IoU-style loss on evidence pointers (door/room IDs mapped to a fixed-size vector).
- **DPO for abstention (Stage C)**: Preference-based objective to encourage abstaining when evidence is missing/inadequate.

### Mathematical Form (Simplified)
- Let `z_s` be student answer logits, `z_t` teacher logits, `y` target token, `T` temperature.
- **Stage A loss**: `L = λ_ce * CE(z_s, y) + λ_kd * KL(softmax(z_t/T) || softmax(z_s/T)) * T^2`
- **Stage B loss**: `L_stageB = L_stageA + λ_e * ||p_s - p_t||_2^2` where `p_*` are evidence pointers.
- **Stage C loss (DPO-style)**: For positive vs. negative preference log-probs `(π_pos, π_neg)` vs reference `(ρ_pos, ρ_neg)`, `L = -log σ(β * (π_pos - π_neg - ρ_pos + ρ_neg))`.

## Training Stages
- **Stage A (Answer KD)**: `make trainA` — trains on filtered labels for answer prediction + logit distillation.
- **Stage B (Evidence KD)**: `make trainB` — loads Stage A ckpt; trains evidence pointers + answer KD.
- **Stage C (DPO Unknown)**: `make trainC` — loads Stage B ckpt; optimizes abstention behavior.
- **Configs**: `configs/trainA.yaml`, `trainB.yaml`, `trainC.yaml`; model config `configs/model.yaml`.

## Evaluation
- **Metrics**: QA accuracy (exact/lenient), evidence F1/IoU, abstention precision/recall, latency/params; adjacency graph edit distance where applicable.
- **Scripts**: `make eval` runs `eval/evaluate.py`; overlays via `make vis`.
- **Stress tests**: Rotation/scale/JPEG/crops in `eval/stress_tests.py`.

## Make Targets (Pipeline)
- `make data` / `make teacher-all`: Download dataset (CubiCasa5k) and generate labels with teacher.
- `make labels`: Critic filter to `train.filtered.jsonl`.
- `make trainA`, `make trainB`, `make trainC`: Staged KD training.
- `make eval`, `make vis`: Metrics and evidence overlays.

## Design Choices & Rationale
- **Pretrained vision/text**: Faster convergence, better priors; freeze vision to reduce compute; small LM for lightweight decoding.
- **Three-stage KD**: Stabilizes learning; separate signal for evidence; final DPO to handle abstention.
- **Multiple questions per image**: Increases label efficiency and variation per sample.
- **Traceability**: `image_id`, `image_path`, `question_key`, `qid` in every record for auditing.
- **Config-driven**: No hard-coded paths; seeds for determinism; outputs under `outputs/<DATE_TAG>/`.
- **Safety**: Abstain (`unknown`) when insufficient evidence; critic filter rejects ungrounded answers.

## Future Improvements
- Swap text model to Phi-2/TinyLlama with QLoRA for parameter-efficient finetuning.
- Incorporate vector/CAD tokens and geometry tools directly into model input or retrieval.
- Add proper image masking/heatmap supervision for evidence IoU.
- Unfreeze top vision blocks or add lightweight adapters for domain shift.
- Expand DPO with explicit positive/negative pairs mined from teacher rationales/tool traces.
