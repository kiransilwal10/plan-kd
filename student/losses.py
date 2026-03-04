"""Loss utilities for staged KD."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def answer_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, temperature: float = 2.0, ce_weight: float = 1.0, kd_weight: float = 1.0) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, targets)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    return ce_weight * ce + kd_weight * kd


def evidence_pointer_loss(student_ptr: torch.Tensor, teacher_ptr: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_ptr, teacher_ptr)


def dpo_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    # Following standard DPO objective: log sigma(beta*(pi_pos - pi_neg - ref_pos + ref_neg))
    diff = beta * (policy_logps - ref_logps)
    return -F.logsigmoid(diff).mean()


def grpo_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: torch.Tensor,
    clip_eps: float = 0.2,
    kl_weight: float = 0.01,
) -> torch.Tensor:
    """Group Relative Policy Optimization loss.

    Args:
        log_probs:     (B, G) log probs of sampled actions under current policy.
        rewards:       (B, G) scalar rewards per rollout.
        ref_log_probs: (B, G) log probs under frozen reference model.
        clip_eps:      PPO-style ratio clipping range.
        kl_weight:     Weight on KL penalty term.
    """
    advantage = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
    ratio = torch.exp(log_probs - log_probs.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    kl = (log_probs.detach() - ref_log_probs).mean()
    return policy_loss + kl_weight * kl


def unpack_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "answer_logits": outputs.get("answer_logits"),
        "evidence_logits": outputs.get("evidence_logits"),
        "abstain_logit": outputs.get("abstain_logit"),
    }
