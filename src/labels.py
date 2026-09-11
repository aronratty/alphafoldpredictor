from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data


LabelTable = dict[str, dict[int, int]]


def load_label_csv(path: str | Path) -> LabelTable:
    """
    Read residue labels.

    Required CSV columns:
        accession,residue_index,label

    residue_index is 1-based sequence order in the AlphaFold graph.
    label must be 0 or 1.

    IMPORTANT:
    Missing rows are treated as UNKNOWN, not as negative labels.
    """
    table: dict[str, dict[int, int]] = defaultdict(dict)

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "residue_index", "label"}

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Label CSV must contain columns: accession,residue_index,label"
            )

        for row in reader:
            accession = row["accession"].strip().upper()
            residue_index = int(row["residue_index"])
            label = int(row["label"])

            if residue_index < 1:
                raise ValueError("residue_index must be 1-based and >= 1")
            if label not in (0, 1):
                raise ValueError("label must be 0 or 1")

            table[accession][residue_index] = label

    return dict(table)


def attach_labels(graph: Data, labels: dict[int, int]) -> Data:
    """
    Add graph.y and graph.label_mask.

    Unspecified residues get y=0 as a placeholder but label_mask=False,
    so they never contribute to training or evaluation.
    """
    n = graph.num_nodes
    assert n is not None
    y = torch.zeros(n, dtype=torch.float32)
    label_mask = torch.zeros(n, dtype=torch.bool)

    for residue_index, label in labels.items():
        zero_based = residue_index - 1
        if zero_based >= n:
            raise IndexError(
                f"Label residue {residue_index} exceeds graph length {n}"
            )

        y[zero_based] = float(label)
        label_mask[zero_based] = True

    graph.y = y
    graph.label_mask = label_mask

    return graph
