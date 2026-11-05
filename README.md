# AlphaFold Protein Binding-Site Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-PyG-EE4C2C?logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Residue-level binding-site prediction from predicted protein structure. Give it a
UniProt accession; it pulls the AlphaFold structure, turns the protein into a 3D
residue graph, and runs a graph attention network that scores every residue.

> **Status:** the structure pipeline, graph construction, model, training loop,
> metrics, and caching are implemented and run end to end. Not done yet: a
> curated experimental label set and a benchmark on homology-safe splits, so
> there are no accuracy numbers to report.

## How it works

```text
UniProt accession
  -> AlphaFold DB REST API -> mmCIF structure
  -> one node per residue (C-alpha coordinates)
  -> edges between residues within 8 Å, distance as an edge feature
  -> 2-layer GAT (4 attention heads, then 1) -> MLP -> one logit per residue
```

Why a graph: residues hundreds of positions apart in sequence can sit next to
each other once the protein folds. A graph built from 3D distance lets each
residue see its real spatial neighbors, not just its sequence neighbors.

**Node features (23):** amino-acid one-hot (21), AlphaFold pLDDT confidence (1),
relative sequence position (1).

**Model:** `GATConv` (4 heads) -> LayerNorm -> ELU -> Dropout -> `GATConv` ->
LayerNorm -> ELU -> MLP classifier.

## Design decisions

- **Class imbalance.** Binding residues are rare (a model that predicts "never
  binds" can score 96% accuracy). Loss is `BCEWithLogitsLoss` with
  `pos_weight = negatives / positives`.
- **Metrics that survive imbalance.** Reports precision, recall, F1, average
  precision, and ROC-AUC; the best checkpoint is chosen on validation average
  precision, not accuracy.
- **Partial labels.** A `label_mask` keeps unannotated residues out of the loss
  instead of treating them as negatives.
- **Leakage.** Residue-level random splits leak between train and test. The
  manifest assigns whole proteins to splits; the next step is clustering by
  sequence similarity so homologs never cross splits.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end smoke test on hemoglobin (untrained weights; scores are not predictions)
python demo.py --accession P69905

# Training (needs a manifest and residue labels; templates in data/)
python train_model.py --manifest data/manifest.csv --labels data/labels.csv --epochs 50
```

Runs on CUDA, Apple MPS, or CPU, whichever is available.

## Repo map

| file | role |
|---|---|
| `src/alphafold_io.py` | UniProt accession -> AlphaFold REST API -> local structure file |
| `src/protein_graph.py` | structure parsing, residue features, distance graph, PyG `Data` |
| `src/labels.py` | attaches residue labels and `label_mask` |
| `src/dataset.py` | lazy download, graph building, and on-disk caching |
| `src/model.py` | the GAT residue classifier |
| `src/training.py` | training and evaluation loops, class weights, metrics |
| `train_model.py` | CLI: data loaders, AdamW, checkpointing, test evaluation |
| `demo.py` | downloads one structure and runs a forward pass |

## Next steps

1. Build an experimental binding-site label set (curated from structures with bound ligands).
2. Cluster proteins by sequence similarity and split by cluster.
3. Benchmark against a residue-only MLP and a GCN to measure what 3D attention adds.
4. Add richer features: solvent accessibility, secondary structure, conservation, protein language model embeddings.

The full write-up, covering every feature, the PyG data layout, and the label pipeline, is in
[docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md).
