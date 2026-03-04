"""Evaluation harness for QA + evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import DataLoader
import yaml
from transformers import AutoTokenizer

from eval.metrics import qa_accuracy, evidence_f1, abstention_stats
from student.datamodules import UnifiedSampleDataset, collate
from student.models.tiny_vlm import PretrainedVLM, PretrainedVLMConfig
from PIL import Image


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
    parser = argparse.ArgumentParser(description="Evaluate tiny VLM.")
    parser.add_argument("--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    data_cfg = cfg if "splits" in cfg else yaml.safe_load(pathlib.Path(cfg["data_cfg"]).read_text())
    test_path = pathlib.Path(data_cfg["splits"]["test"])
    dataset = UnifiedSampleDataset(test_path)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer_name = ckpt.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_cfg = PretrainedVLMConfig(**ckpt.get("config", {}))
    model = PretrainedVLM(model_cfg, tokenizer=tokenizer)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)

    preds = []
    refs = []
    evid_preds = []
    evid_refs = []
    abstain_preds = []
    abstain_refs = []
    latencies = []
    max_len = model_cfg.max_length if hasattr(model_cfg, "max_length") else 128

    with torch.no_grad():
        for batch in loader:
            enc = tokenizer(
                batch["questions"],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            image_paths = [m.get("image_path", "") for m in batch.get("meta", [])]
            pixel_values = load_pixels(image_paths, model.vision_processor, device)
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            latencies.append(time.time() - start)

            top_ids = outputs["answer_logits"].argmax(dim=-1).cpu().tolist()
            batch_preds = [tokenizer.decode([idx]).strip() for idx in top_ids]
            preds.extend(batch_preds)
            refs.extend(batch["answers"])

            evid_pred_ids = outputs["evidence_logits"].topk(k=3, dim=-1).indices.cpu().tolist()
            evid_preds.extend([[f"id-{i}" for i in ids] for ids in evid_pred_ids])
            evid_refs.extend([ev.get("door_ids", []) + ev.get("room_ids", []) for ev in batch["evidence"]])

            abstain = (outputs["abstain_logit"].sigmoid() > 0.5).cpu().tolist()
            abstain_preds.extend(abstain)
            abstain_refs.extend([ans.strip().lower() == "unknown" for ans in batch["answers"]])

    metrics = {
        "qa": qa_accuracy(preds, refs),
        "evidence_f1": evidence_f1(evid_preds, evid_refs),
        "abstention": abstention_stats(abstain_preds, abstain_refs),
        "latency_ms": 1000 * sum(latencies) / max(len(latencies), 1),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("outputs") / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = pathlib.Path("reports/tables")
    report_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out_dir/'metrics.json'} and {report_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
