"""Stage B: Evidence distillation (pointer/IoU)."""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml
from transformers import AutoTokenizer
from PIL import Image

from student.datamodules import UnifiedSampleDataset, collate
from student.losses import answer_kd_loss, evidence_pointer_loss
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage B (evidence KD).")
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

    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"]["num_workers"],
        collate_fn=collate,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    max_len = model_cfg.max_length if hasattr(model_cfg, "max_length") else 128

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
            targets = torch.tensor(
                [
                    (tokenizer(ans, add_special_tokens=False).input_ids[:1] or [0])[0]
                    for ans in answers
                ],
                device=device,
            )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            teacher_logits = outputs["answer_logits"].detach()
            ans_loss = answer_kd_loss(
                outputs["answer_logits"],
                teacher_logits,
                targets,
                temperature=cfg["loss"]["temperature"],
                ce_weight=cfg["loss"]["ce_weight"],
                kd_weight=cfg["loss"]["kd_weight"],
            )
            teacher_ptr = torch.stack([evidence_vector(ev) for ev in evidence]).to(device)
            ptr_loss = evidence_pointer_loss(outputs["evidence_logits"], teacher_ptr)
            loss = ans_loss + cfg["loss"]["evidence_weight"] * ptr_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % cfg["trainer"]["log_interval"] == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(cfg.get("output_root", "outputs")) / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student-B.pt"
    torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.name_or_path, "config": model_cfg.__dict__}, ckpt_path)
    print(f"Saved Stage B checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
