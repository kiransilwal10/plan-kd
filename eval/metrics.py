"""Evaluation metrics for floor-plan QA + evidence."""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import networkx as nx
import numpy as np


def qa_accuracy(preds: List[str], refs: List[str]) -> Dict[str, float]:
    exact = sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)
    lenient = sum(p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs)) / max(len(refs), 1)
    return {"exact": exact, "lenient": lenient}


def evidence_f1(pred_ids: List[List[str]], ref_ids: List[List[str]]) -> float:
    tp = fp = fn = 0
    for pred, ref in zip(pred_ids, ref_ids):
        pset, rset = set(pred), set(ref)
        tp += len(pset & rset)
        fp += len(pset - rset)
        fn += len(rset - pset)
    denom = max(2 * tp + fp + fn, 1)
    return (2 * tp) / denom


def iou_masks(pred_masks: List[np.ndarray], ref_masks: List[np.ndarray]) -> float:
    scores = []
    for pm, rm in zip(pred_masks, ref_masks):
        if pm is None or rm is None:
            continue
        inter = np.logical_and(pm, rm).sum()
        union = np.logical_or(pm, rm).sum()
        scores.append(inter / union if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def graph_edit_distance(pred: nx.Graph, ref: nx.Graph) -> float:
    try:
        return nx.graph_edit_distance(pred, ref)
    except Exception:
        return float("inf")


def abstention_stats(pred_unknown: List[bool], ref_unknown: List[bool]) -> Dict[str, float]:
    tp = sum(p and r for p, r in zip(pred_unknown, ref_unknown))
    fp = sum(p and not r for p, r in zip(pred_unknown, ref_unknown))
    fn = sum((not p) and r for p, r in zip(pred_unknown, ref_unknown))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall}
