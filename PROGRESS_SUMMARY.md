# Honors Thesis Progress Summary
## Distilling a Large Vision-Language Model into a Compact Student for Evidence-Grounded Floor Plan Question Answering

**Student:** Kiran Silwal
**Date:** March 2026

---

## 1. Project Overview

My honors thesis investigates **knowledge distillation** for architectural floor plan understanding. The core problem: large vision-language models (e.g., GPT-4o) can answer complex questions about floor plans, but they are too expensive, slow, and closed-source for practical deployment. My research builds a pipeline that transfers this capability into a **compact, open student model** that can run locally and efficiently.

The student model not only answers questions but also **points to supporting evidence** (specific doors and rooms) and **abstains when uncertain** — a safety-critical behavior for architectural analysis.

---

## 2. What Has Been Built

The full codebase consists of **~2,080 lines of Python** across 20+ modules, organized into four major components:

### Repository Structure
```
plan-kd/
├── teacher/          # Label generation via GPT-4o + critic filtering
├── student/          # Compact VLM model, losses, 3-stage training
├── eval/             # Metrics, evaluation loop, stress tests, visualization
├── tools/geometry/   # Adjacency graph + spatial utilities
├── scripts/          # Dataset download/parsing pipeline
├── configs/          # YAML configs for model, data, training stages
├── data/             # CubiCasa5k dataset + 37,000 generated labels
└── docs/             # Methodology writeup (Markdown + LaTeX)
```

### Dataset and Labels
- **Dataset:** CubiCasa5k (5,000 residential floor plan images)
- **Generated labels:** 37,027 teacher-generated QA pairs with evidence annotations
- **Question bank:** 70+ diverse question types (door counts, room adjacency, accessibility, egress paths, etc.)
- Each label includes: answer, evidence pointers (door_ids, room_ids), rationale, uncertainty score, and tool traces

---

## 3. Technical Approach: Three-Stage Knowledge Distillation

### Stage A — Answer Distillation
The student learns to replicate the teacher's answers using a combined cross-entropy and KL-divergence loss:

```
L_A = λ_ce * CE(z_s, y) + λ_kd * T² * KL(softmax(z_t/T) || softmax(z_s/T))
```

### Stage B — Evidence Distillation
Building on Stage A, the student learns to point to the same evidence (doors/rooms) the teacher used, via a pointer-matching MSE loss:

```
L_B = L_A + λ_ev * ||p_s - p_t||²
```

### Stage C — GRPO for Abstention and Joint Optimization
Using Group Relative Policy Optimization, the student learns when to answer and when to say "unknown," jointly optimizing answer quality, evidence accuracy, and abstention behavior with a composite reward:

```
Reward = 0.5 * r_answer + 0.3 * r_evidence + 0.2 * r_abstention
```

---

## 4. Key Code Snippets

### 4.1 Student Model Architecture (student/models/tiny_vlm.py)

The compact model fuses a frozen CLIP vision encoder with a small GPT-2 text decoder:

```python
class PretrainedVLM(nn.Module):
    """Simple fusion: pooled vision + pooled text -> projection -> heads."""

    def __init__(self, cfg: PretrainedVLMConfig, tokenizer=None):
        super().__init__()
        # Vision: CLIP ViT-Base (frozen) — encodes floor plan image
        self.vision_model = CLIPVisionModel.from_pretrained(cfg.vision_model)
        # Text: GPT-2 — encodes the question
        self.text_model = AutoModel.from_pretrained(cfg.text_model)

        # Fusion: concatenate vision + text, project, then 3 task heads
        fused_dim = vision_dim + text_dim
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, cfg.projector_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.answer_head = nn.Linear(cfg.projector_hidden, self.tokenizer.vocab_size)
        self.evidence_head = nn.Linear(cfg.projector_hidden, cfg.evidence_dim)
        self.abstain_head = nn.Linear(cfg.projector_hidden, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_repr = self.encode_text(input_ids, attention_mask)
        vision_repr = self.encode_image(pixel_values)
        fused = torch.cat([vision_repr, text_repr], dim=-1)
        fused = self.projector(fused)
        return {
            "answer_logits": self.answer_head(fused),
            "evidence_logits": self.evidence_head(fused),
            "abstain_logit": self.abstain_head(fused).squeeze(-1),
        }
```

### 4.2 Loss Functions (student/losses.py)

```python
def answer_kd_loss(student_logits, teacher_logits, targets, temperature=2.0,
                   ce_weight=1.0, kd_weight=1.0):
    ce = F.cross_entropy(student_logits, targets)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return ce_weight * ce + kd_weight * kd

def grpo_loss(log_probs, rewards, ref_log_probs, clip_eps=0.2, kl_weight=0.01):
    """Group Relative Policy Optimization loss."""
    advantage = (rewards - rewards.mean(dim=1, keepdim=True)) / \
                (rewards.std(dim=1, keepdim=True) + 1e-8)
    ratio = torch.exp(log_probs - log_probs.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    kl = (log_probs.detach() - ref_log_probs).mean()
    return policy_loss + kl_weight * kl
```

### 4.3 Teacher Label Generation (teacher/generate_labels.py)

The teacher pipeline sends floor plan images to GPT-4o-mini and collects structured labels:

```python
def call_teacher(client, model, system_prompt, user_text, image_data_url, ...):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]},
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)
```

### 4.4 GRPO Reward Computation (student/train_stageC_dpo_unknown.py)

```python
def compute_rewards(sampled_ids, evidence_logits, abstain_logit,
                    gold_answers, gold_evidence, tokenizer, reward_cfg):
    """Composite reward: answer accuracy + evidence F1 + abstention correctness."""
    for i in range(B):
        # Answer: exact match
        r_ans = 1.0 if pred == gold else 0.0
        # Evidence: element-wise F1 on binary pointer vector
        r_ev = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
        # Abstention: correct abstain/answer decision
        r_abs = 1.0 if pred_abstain == should_abstain else 0.0
        rewards[i] = w_ans * r_ans + w_ev * r_ev + w_abs * r_abs
    return rewards
```

### 4.5 Evaluation Metrics (eval/metrics.py)

```python
def qa_accuracy(preds, refs):
    exact = sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)
    lenient = sum(p.strip().lower() == r.strip().lower()
                  for p, r in zip(preds, refs)) / max(len(refs), 1)
    return {"exact": exact, "lenient": lenient}

def evidence_f1(pred_ids, ref_ids):
    # Standard F1 over predicted vs. gold evidence sets
    ...

def abstention_stats(pred_unknown, ref_unknown):
    # Precision and recall on "unknown" decisions
    ...
```

---

## 5. Sample Generated Label

Each teacher-generated label looks like this (from the 37,027 in the dataset):

```json
{
  "qid": "cubicasa5k/colorful/10052/F1_original.png-q0",
  "question": "Do doorways/hallways appear wide enough for wheelchair access?",
  "answer": "yes",
  "evidence": {
    "door_ids": ["main entrance door", "bathroom door", "other internal doors"],
    "room_ids": ["kitchen", "bathroom", "living areas"]
  },
  "notes": "The doorways and hallways appear wide enough for wheelchair
            access based on the proportions shown...",
  "uncertainty": 0.0
}
```

---

## 6. Current Status and Next Steps

### Completed
- Full project architecture and codebase (~2,080 lines of Python)
- Teacher label generation pipeline (37,027 QA pairs with evidence)
- Critic filter for quality control
- Student model architecture (CLIP + GPT-2 fusion with 3 task heads)
- All three training stages implemented (Answer KD, Evidence KD, GRPO)
- Evaluation harness with metrics (QA accuracy, evidence F1, abstention stats)
- Robustness stress tests (rotation, scale, JPEG compression, cropping)
- Evidence visualization pipeline
- LaTeX methodology writeup

### In Progress / Next Steps
- Run full-scale training on GPU cluster and collect quantitative results
- Compare student vs. teacher accuracy and evidence quality
- Ablation studies across training stages (Stage A only vs. A+B vs. A+B+C)
- Explore swapping GPT-2 for a stronger small LM (Phi-2 or TinyLlama)
- Write final thesis document with results and analysis

---

## 7. Pipeline (Makefile)

The entire workflow is reproducible via Make targets:

```
make data       # Download CubiCasa5k
make teacher    # Generate teacher labels via GPT-4o
make labels     # Critic filter
make trainA     # Stage A: Answer distillation
make trainB     # Stage B: Evidence distillation
make trainC     # Stage C: GRPO optimization
make eval       # Evaluate on test set
make vis        # Generate evidence overlay visualizations
```
