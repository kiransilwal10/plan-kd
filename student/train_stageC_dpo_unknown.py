"""Stage C: GRPO finetuning for joint answer, evidence, and abstention quality."""

from __future__ import annotations

import argparse
import copy
import pathlib
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from transformers import AutoTokenizer
from PIL import Image

from student.datamodules import UnifiedSampleDataset, collate
from student.losses import grpo_loss
from student.models.tiny_vlm import PretrainedVLM, PretrainedVLMConfig


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pixels(image_paths, processor, device):
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (processor.size["shortest_edge"], processor.size["shortest_edge"]), color=(255, 255, 255))
        images.append(img)
    batch = processor(images=images, return_tensors="pt")
    return batch["pixel_values"].to(device)


def evidence_vector(evidence: dict, length: int = 32) -> torch.Tensor:
    vec = torch.zeros(length)
    for idx, _ in enumerate(evidence.get("door_ids", []) + evidence.get("room_ids", [])):
        if idx < length:
            vec[idx] = 1.0
    return vec


def compute_rewards(
    sampled_ids: torch.Tensor,
    evidence_logits: torch.Tensor,
    abstain_logit: torch.Tensor,
    gold_answers: List[str],
    gold_evidence: List[dict],
    tokenizer,
    reward_cfg: dict,
) -> torch.Tensor:
    """Compute composite reward for each sample in the batch.

    Reward = answer_weight * r_ans
           + evidence_weight * r_ev
           + abstain_weight * r_abs
    """
    B = sampled_ids.shape[0]
    rewards = torch.zeros(B, device=sampled_ids.device)

    w_ans = reward_cfg.get("answer_weight", 0.5)
    w_ev = reward_cfg.get("evidence_weight", 0.3)
    w_abs = reward_cfg.get("abstain_weight", 0.2)

    for i in range(B):
        # Answer reward: exact match (case-insensitive, stripped)
        pred = tokenizer.decode([sampled_ids[i].item()]).strip().lower()
        gold = gold_answers[i].strip().lower()
        r_ans = 1.0 if pred == gold else 0.0

        # Evidence reward: element-wise F1 on the 32-dim binary vector
        pred_ev = (evidence_logits[i] > 0.0).float().cpu()
        gold_ev = evidence_vector(gold_evidence[i], length=evidence_logits.shape[-1])
        tp = (pred_ev * gold_ev).sum().item()
        fp = (pred_ev * (1.0 - gold_ev)).sum().item()
        fn = ((1.0 - pred_ev) * gold_ev).sum().item()
        r_ev = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)

        # Abstention reward: correct abstain/answer decision
        should_abstain = not (gold_evidence[i].get("door_ids") or gold_evidence[i].get("room_ids"))
        pred_abstain = torch.sigmoid(abstain_logit[i]).item() > 0.5
        r_abs = 1.0 if pred_abstain == should_abstain else 0.0

        rewards[i] = w_ans * r_ans + w_ev * r_ev + w_abs * r_abs

    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage C (GRPO).")
    parser.add_argument("--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    set_seed(cfg.get("seed", 42))
    data_cfg = yaml.safe_load(pathlib.Path(cfg["data_cfg"]).read_text())
    train_path = pathlib.Path(data_cfg["splits"]["train"])
    dataset = UnifiedSampleDataset(train_path)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer_name = ckpt.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cfg = PretrainedVLMConfig(**ckpt.get("config", {}))
    model = PretrainedVLM(model_cfg, tokenizer=tokenizer)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(cfg["trainer"].get("device", "cpu"))
    model.to(device)

    # Frozen reference model (Stage B checkpoint — never updated)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"]["num_workers"],
        collate_fn=collate,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["grpo"].get("lr", 1e-5))
    max_len = model_cfg.max_length if hasattr(model_cfg, "max_length") else 128
    G = cfg["grpo"]["group_size"]
    clip_eps = cfg["grpo"].get("clip_eps", 0.2)
    kl_weight = cfg["grpo"].get("kl_weight", 0.01)
    reward_cfg = cfg["grpo"].get("reward", {})

    for epoch in range(cfg["trainer"]["epochs"]):
        for step, batch in enumerate(loader):
            questions = batch["questions"]
            answers = batch["answers"]
            evidence = batch["evidence"]

            enc = tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            image_paths = [m.get("image_path", "") for m in batch.get("meta", [])]
            pixel_values = load_pixels(image_paths, model.vision_processor, device)

            # Pre-compute gold targets for evidence and abstain heads (shared across rollouts)
            ev_dim = model_cfg.evidence_dim
            gold_ev = torch.stack([evidence_vector(ev, length=ev_dim) for ev in evidence]).to(device)  # (B, ev_dim)
            should_abstain = torch.tensor(
                [0.0 if (ev.get("door_ids") or ev.get("room_ids")) else 1.0 for ev in evidence],
                device=device,
            )  # (B,)

            w_ans = reward_cfg.get("answer_weight", 0.5)
            w_ev  = reward_cfg.get("evidence_weight", 0.3)
            w_abs = reward_cfg.get("abstain_weight", 0.2)

            # Collect G rollouts
            all_log_probs = []
            all_rewards = []
            all_ref_log_probs = []

            model.train()
            for _ in range(G):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

                # Gumbel sampling over answer vocab
                gumbel = -torch.log(-torch.log(torch.rand_like(outputs["answer_logits"]).clamp(min=1e-10)) + 1e-10)
                sampled_ids = (outputs["answer_logits"] + gumbel).argmax(dim=-1)  # (B,)

                # Composite log prob: answer + evidence (Bernoulli per dim) + abstain (Bernoulli)
                lp_ans = F.log_softmax(outputs["answer_logits"], dim=-1).gather(
                    1, sampled_ids.unsqueeze(1)
                ).squeeze(1)  # (B,)
                lp_ev = F.logsigmoid(
                    outputs["evidence_logits"] * (2.0 * gold_ev - 1.0)
                ).sum(dim=-1) / ev_dim  # (B,) normalized
                lp_abs = F.logsigmoid(
                    outputs["abstain_logit"] * (2.0 * should_abstain - 1.0)
                )  # (B,)
                log_probs_g = w_ans * lp_ans + w_ev * lp_ev + w_abs * lp_abs  # (B,)

                with torch.no_grad():
                    ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                    rlp_ans = F.log_softmax(ref_out["answer_logits"], dim=-1).gather(
                        1, sampled_ids.unsqueeze(1)
                    ).squeeze(1)
                    rlp_ev = F.logsigmoid(
                        ref_out["evidence_logits"] * (2.0 * gold_ev - 1.0)
                    ).sum(dim=-1) / ev_dim
                    rlp_abs = F.logsigmoid(
                        ref_out["abstain_logit"] * (2.0 * should_abstain - 1.0)
                    )
                    ref_log_probs_g = w_ans * rlp_ans + w_ev * rlp_ev + w_abs * rlp_abs  # (B,)

                rewards_g = compute_rewards(
                    sampled_ids,
                    outputs["evidence_logits"].detach(),
                    outputs["abstain_logit"].detach(),
                    answers,
                    evidence,
                    tokenizer,
                    reward_cfg,
                )

                all_log_probs.append(log_probs_g)
                all_rewards.append(rewards_g)
                all_ref_log_probs.append(ref_log_probs_g)

            # Stack to (B, G)
            log_probs_bg = torch.stack(all_log_probs, dim=1)
            rewards_bg = torch.stack(all_rewards, dim=1)
            ref_log_probs_bg = torch.stack(all_ref_log_probs, dim=1)

            loss = grpo_loss(log_probs_bg, rewards_bg, ref_log_probs_bg, clip_eps=clip_eps, kl_weight=kl_weight)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % cfg["trainer"]["log_interval"] == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(cfg.get("output_root", "outputs")) / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student-C.pt"
    torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.name_or_path, "config": model_cfg.__dict__}, ckpt_path)
    print(f"Saved Stage C checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
