from __future__ import annotations

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    positives = labels == 1
    tp = int(np.logical_and(predictions, positives).sum())
    tn = int(np.logical_and(~predictions, ~positives).sum())
    fp = int(np.logical_and(predictions, ~positives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auroc": roc_auc(labels, scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    tp = 0
    fp = 0
    best_value = 0.5
    best_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        recall = tp / n_pos
        specificity = (n_neg - fp) / n_neg
        value = 0.5 * (recall + specificity)
        if value > best_value:
            best_value = value
            best_threshold = float(sorted_scores[start])
        start = end
    return best_threshold
