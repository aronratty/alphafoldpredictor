from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch_geometric.data import Dataset, Data

from .alphafold_io import download_structure
from .labels import LabelTable, attach_labels, load_label_csv
from .protein_graph import structure_to_graph


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None or not {
            "accession",
            "split"
        }.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest CSV must contain columns: accession,split"
            )

        for row in reader:
            rows.append({
                "accession": row["accession"].strip().upper(),
                "split": row["split"].strip().lower(),
            })

    return rows


class ProteinGraphDataset(Dataset):

    def __init__(
        self,
        manifest_csv: str | Path,
        labels_csv: str | Path,
        *,
        split: str,
        cache_dir: str | Path = "cache",
        cutoff_angstrom: float = 8.0,
    ):
        self.split = split.lower()

        self.cache_dir = Path(cache_dir)
        self.structure_dir = self.cache_dir / "structures"
        self.graph_dir = self.cache_dir / "graphs"

        self.structure_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.graph_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        manifest = load_manifest(manifest_csv)

        self.accessions = [
            row["accession"]
            for row in manifest
            if row["split"] == self.split
        ]

        self.label_table: LabelTable = load_label_csv(
            labels_csv
        )

        self.cutoff_angstrom = float(
            cutoff_angstrom
        )

        # Important for torch_geometric.data.Dataset
        super().__init__()

    def len(self) -> int:
        return len(self.accessions)

    def get(self, idx: int) -> Data:

        accession = self.accessions[idx]

        cache_path = (
            self.graph_dir
            / f"{accession}_r{self.cutoff_angstrom:g}.pt"
        )

        if cache_path.exists():
            return torch.load(
                cache_path,
                weights_only=False
            )

        structure_path, _ = download_structure(
            accession,
            self.structure_dir,
            fmt="cif",
        )

        graph = structure_to_graph(
            structure_path,
            accession=accession,
            cutoff_angstrom=self.cutoff_angstrom,
        )

        labels = self.label_table.get(
            accession,
            {}
        )

        graph = attach_labels(
            graph,
            labels
        )

        torch.save(
            graph,
            cache_path
        )

        return graph