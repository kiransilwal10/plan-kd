"""Overlay predicted evidence on floor-plan images."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from student.datamodules import UnifiedSampleDataset
from student.models.tiny_vlm import PretrainedVLM, PretrainedVLMConfig
from eval.evaluate import load_pixels


def draw_overlay(idx: int, answer: str, evid_ids: list[str], out_dir: pathlib.Path) -> None:
    canvas = Image.new("RGB", (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((10, 10, 502, 502), outline=(80, 80, 80), width=2)
    draw.text((20, 20), f"Q{idx} ans={answer}", fill=(0, 0, 0))
    for j, evid in enumerate(evid_ids):
        draw.ellipse((50 + j * 40, 200, 70 + j * 40, 220), fill=(200, 50, 50))
        draw.text((45 + j * 40, 225), evid, fill=(0, 0, 0))
    canvas.save(out_dir / f"sample_{idx:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize evidence overlays.")
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--cfg", type=pathlib.Path, default=pathlib.Path("configs/data.yaml"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = UnifiedSampleDataset(pathlib.Path("data/labels") / f"{args.split}.filtered.jsonl")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer_name = ckpt.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_cfg = PretrainedVLMConfig(**ckpt.get("config", {}))
    model = PretrainedVLM(model_cfg, tokenizer=tokenizer)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_len = model_cfg.max_length if hasattr(model_cfg, "max_length") else 64
    with torch.no_grad():
        for idx, sample in enumerate(dataset.samples[: args.max]):
            enc = tokenizer(
                [sample["question"]],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            image_paths = [sample.get("image_path", "")]
            pixel_values = load_pixels(image_paths, model.vision_processor, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            pred_idx = outputs["answer_logits"].argmax(dim=-1).item()
            answer = tokenizer.decode([pred_idx]).strip()
            evid_ids = [f"id-{i}" for i in outputs["evidence_logits"].topk(k=3, dim=-1).indices.cpu().tolist()[0]]
            draw_overlay(idx, answer, evid_ids, args.out)

    print(f"Saved overlays to {args.out}")


if __name__ == "__main__":
    main()
