from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.alphafold_io import download_structure
from src.model import ProteinBindingGAT
from src.protein_graph import NODE_FEATURE_DIM, structure_to_graph


def main():
    parser = argparse.ArgumentParser(
        description="Fetch an AlphaFold structure and run an untrained GAT forward pass."
    )
    parser.add_argument("--accession", default="P69905")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    accession = args.accession.strip().upper()
    structure_dir = Path(args.cache_dir) / "structures"

    print(f"Fetching AlphaFold structure for {accession} ...")
    structure_path, metadata = download_structure(
        accession,
        structure_dir,
        fmt="cif",
    )

    graph = structure_to_graph(
        structure_path,
        accession=accession,
        cutoff_angstrom=args.cutoff,
    )

    print(f"Structure: {structure_path}")
    print(f"Residues/nodes: {graph.num_nodes}")
    print(f"Directed graph edges: {graph.num_edges}")
    assert graph.x is not None
    print(f"Node feature shape: {tuple(graph.x.shape)}")
    assert graph.pos is not None
    print(f"Coordinates shape: {tuple(graph.pos.shape)}")
    assert graph.edge_attr is not None
    print(f"Edge attribute shape: {tuple(graph.edge_attr.shape)}")

    model = ProteinBindingGAT(
        input_dim=NODE_FEATURE_DIM,
        hidden_dim=64,
        heads=4,
    )
    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(graph.x, graph.edge_index, graph.edge_attr)
        )

    assert graph.num_nodes is not None 
    top_k = min(10, graph.num_nodes)
    values, indices = torch.topk(probabilities, k=top_k)

    print("\nWARNING: The model is untrained. These are not biological predictions.")
    print("Top random model outputs:")
    for score, zero_based_index in zip(values.tolist(), indices.tolist()):
        residue_name = graph.residue_names[zero_based_index]
        sequence_index = zero_based_index + 1
        print(
            f"  residue {sequence_index:4d} "
            f"{residue_name:>3s}: probability={score:.4f}"
        )

    model_id = metadata.get("modelEntityId", metadata.get("entryId", "unknown"))
    print(f"\nAlphaFold model entity: {model_id}")


if __name__ == "__main__":
    main()
