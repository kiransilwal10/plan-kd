"""Stage A: Answer distillation (CE + logit KD)."""

from __future__ import annotations

import argparse
import pathlib
import random
from datetime import datetime
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import yaml
from transformers import AutoTokenizer
from PIL import Image

from student.datamodules import UnifiedSampleDataset, collate
from student.losses import answer_kd_loss
from student.models.tiny_vlm import PretrainedVLM, PretrainedVLMConfig


def set_seed(seed: int) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage A (answer KD).")
    parser.add_argument("--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    set_seed(cfg.get("seed", 42))

    data_cfg = yaml.safe_load(pathlib.Path(cfg["data_cfg"]).read_text())
    model_cfg_raw = yaml.safe_load(pathlib.Path(cfg["model_cfg"]).read_text())
    train_path = pathlib.Path(data_cfg["splits"]["train"])
    train_dataset = UnifiedSampleDataset(train_path)

    vision_model = model_cfg_raw["model"].get("vision_model", "openai/clip-vit-base-patch32")
    text_model = model_cfg_raw["model"].get("text_model", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cfg = PretrainedVLMConfig(
        vision_model=vision_model,
        text_model=text_model,
        max_length=model_cfg_raw["model"].get("max_length", 128),
        projector_hidden=model_cfg_raw["model"].get("projector_hidden", 512),
        evidence_dim=model_cfg_raw["model"].get("evidence_dim", 32),
        freeze_vision=model_cfg_raw["model"].get("freeze_vision", True),
        freeze_text=model_cfg_raw["model"].get("freeze_text", False),
    )
    model = PretrainedVLM(model_cfg, tokenizer=tokenizer)
    device = torch.device(cfg["trainer"].get("device", "cpu"))
    model.to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"]["num_workers"],
        collate_fn=collate,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr_adapters"] if "optim" in cfg else 1e-4)
    max_len = model_cfg.max_length if hasattr(model_cfg, "max_length") else 128

    for epoch in range(cfg["trainer"]["epochs"]):
        for step, batch in enumerate(loader):
            questions = batch["questions"]
            answers = batch["answers"]
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
            teacher_logits = outputs["answer_logits"].detach() + 0.01 * torch.randn_like(outputs["answer_logits"])
            loss = answer_kd_loss(
                outputs["answer_logits"],
                teacher_logits,
                targets,
                temperature=cfg["loss"]["temperature"],
                ce_weight=cfg["loss"]["ce_weight"],
                kd_weight=cfg["loss"]["kd_weight"],
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % cfg["trainer"]["log_interval"] == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(cfg.get("output_root", "outputs")) / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student-A.pt"
    torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.name_or_path, "config": model_cfg.__dict__}, ckpt_path)
    print(f"Saved Stage A checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
