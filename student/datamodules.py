"""Data modules for student training."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class UnifiedSampleDataset(Dataset):
    def __init__(self, labels_path: pathlib.Path):
        self.samples = self._load_jsonl(labels_path)

    def _load_jsonl(self, path: pathlib.Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        sample = {
            "qid": record.get("qid"),
            "image_id": record.get("image_id"),
            "image_path": record.get("image_path"),
            "question_key": record.get("question_key"),
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "evidence": record.get("evidence", {}),
            "notes": record.get("notes", ""),
            "must_ground": record.get("must_ground", False),
            "uncertainty": record.get("uncertainty", 0.0),
        }
        return sample


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    evidence = [item["evidence"] for item in batch]
    return {
        "questions": questions,
        "answers": answers,
        "evidence": evidence,
        "meta": [
            {
                "qid": item.get("qid"),
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "question_key": item.get("question_key"),
            }
            for item in batch
        ],
    }


def build_dataloaders(cfg_path: pathlib.Path, split: str, batch_size: int, num_workers: int) -> DataLoader:
    cfg = yaml.safe_load(cfg_path.read_text())
    split_path = pathlib.Path(cfg["splits"][split])
    dataset = UnifiedSampleDataset(split_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=split == "train", collate_fn=collate)
