from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    loss: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float | None


def _labeled_logits_and_targets(batch, logits: torch.Tensor):
    mask = batch.label_mask.bool()
    return logits[mask], batch.y[mask].float()


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_labels = 0

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
        )
        labeled_logits, targets = _labeled_logits_and_targets(batch, logits)

        if targets.numel() == 0:
            continue

        loss = loss_fn(labeled_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * targets.numel()
        total_labels += targets.numel()

    if total_labels == 0:
        raise RuntimeError(
            "Training set contained zero labeled residues. "
            "Populate labels.csv and ensure manifest accessions match."
        )

    return total_loss / total_labels


@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> Metrics:
    model.eval()

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_labels = 0

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
        )

        labeled_logits, targets = _labeled_logits_and_targets(batch, logits)

        if targets.numel() == 0:
            continue

        loss = loss_fn(labeled_logits, targets)
        probabilities = torch.sigmoid(labeled_logits)

        total_loss += float(loss.item()) * targets.numel()
        total_labels += targets.numel()

        all_probabilities.append(probabilities.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    if total_labels == 0:
        raise RuntimeError("Evaluation split contained zero labeled residues.")

    y_prob = np.concatenate(all_probabilities)
    y_true = np.concatenate(all_targets).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    # zero_division=0 keeps metrics well-defined for tiny/demo datasets.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    average_precision = average_precision_score(y_true, y_prob)

    roc_auc: float | None
    if np.unique(y_true).size == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = None

    return Metrics(
        loss=total_loss / total_labels,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=float(average_precision),
        roc_auc=roc_auc,
    )


def compute_pos_weight(dataset) -> float:
    """
    Return negatives / positives over labeled training residues.

    BCEWithLogitsLoss(pos_weight=...) uses this to make rare positive
    binding residues contribute more strongly to the loss.
    """
    positives = 0
    negatives = 0

    for graph in dataset:
        mask = graph.label_mask.bool()
        labels = graph.y[mask]

        positives += int((labels == 1).sum())
        negatives += int((labels == 0).sum())

    if positives == 0:
        raise RuntimeError("No positive binding-site labels in training data.")
    if negatives == 0:
        raise RuntimeError("No negative binding-site labels in training data.")

    return negatives / positives
