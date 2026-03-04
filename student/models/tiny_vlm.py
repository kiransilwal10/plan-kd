"""Pretrained vision + text encoder fusion model (CLIP vision + small LM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)


@dataclass
class PretrainedVLMConfig:
    vision_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "gpt2"
    max_length: int = 128
    projector_hidden: int = 512
    evidence_dim: int = 32
    freeze_vision: bool = True
    freeze_text: bool = False


class PretrainedVLM(nn.Module):
    """Simple fusion: pooled vision + pooled text -> projection -> heads."""

    def __init__(self, cfg: PretrainedVLMConfig, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__()
        self.cfg = cfg

        self.vision_processor = CLIPImageProcessor.from_pretrained(cfg.vision_model)
        vision_cfg = CLIPVisionConfig.from_pretrained(cfg.vision_model)
        self.vision_model = CLIPVisionModel.from_pretrained(cfg.vision_model)

        text_cfg = AutoConfig.from_pretrained(cfg.text_model)
        self.text_model = AutoModel.from_pretrained(cfg.text_model, config=text_cfg)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.text_model)

        vision_dim = vision_cfg.hidden_size
        text_dim = text_cfg.hidden_size
        fused_dim = vision_dim + text_dim

        self.projector = nn.Sequential(
            nn.Linear(fused_dim, cfg.projector_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.answer_head = nn.Linear(cfg.projector_hidden, self.tokenizer.vocab_size)
        self.evidence_head = nn.Linear(cfg.projector_hidden, cfg.evidence_dim)
        self.abstain_head = nn.Linear(cfg.projector_hidden, 1)

        if cfg.freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False
        if cfg.freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (b, seq, hidden)
        # pooled text: take first token or attention-weighted mean
        mask = attention_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output  # (b, hidden)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_repr = self.encode_text(input_ids, attention_mask)
        vision_repr = self.encode_image(pixel_values)
        fused = torch.cat([vision_repr, text_repr], dim=-1)
        fused = self.projector(fused)
        return {
            "answer_logits": self.answer_head(fused),
            "evidence_logits": self.evidence_head(fused),
            "abstain_logit": self.abstain_head(fused).squeeze(-1),
        }
