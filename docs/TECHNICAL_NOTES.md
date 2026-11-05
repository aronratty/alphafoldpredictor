# AlphaFold Protein Binding Site Predictor

A residue-level protein binding-site prediction pipeline built with **PyTorch**, **PyTorch Geometric**, **BioPython**, and the **AlphaFold Protein Structure Database**.

The project takes a protein identifier, retrieves its predicted 3D structure from AlphaFold DB, converts the structure into a residue graph, and runs a graph-attention neural network over that graph to produce one binding-site score per amino-acid residue.

> **Project status:** the structural pipeline, graph construction, model architecture, training loop, evaluation code, and caching system are implemented. The remaining research step is building a trustworthy experimental binding-site label dataset and benchmarking the model on leakage-safe train/validation/test splits.

---

## Overview

```text
UniProt accession
        ↓
AlphaFold DB REST API
        ↓
Predicted protein structure
        ↓
Residue-level features + 3D coordinates
        ↓
Distance-based residue graph
        ↓
Graph Attention Network
        ↓
One binding-site logit per residue
        ↓
Sigmoid
        ↓
Binding probability per residue
```

The central idea is that a residue should not be classified only from its own amino-acid identity. Protein function depends heavily on the residue's **3D environment**.

Two amino acids that are hundreds of positions apart in the primary sequence can become physically adjacent after the protein folds. Representing the protein as a graph lets the model exchange information between residues that are actually close in space.

---

## Problem Formulation

For a protein with `N` residues, the model receives:

```text
N residues
+
features for each residue
+
3D coordinates
+
connections between spatial neighbors
```

and produces `N` logits, one per residue.

The supervised target is:

```text
0 = non-binding residue
1 = binding residue
```

Conceptually:

```text
Residue 1   → 0.03
Residue 2   → 0.11
Residue 3   → 0.87
Residue 4   → 0.91
```

The model therefore performs **node-level binary classification** on a protein graph.

---

## Why 3D Structure Matters

A protein starts as a sequence:

```text
M E T L K A V ...
```

but folds into a three-dimensional structure.

Residues that are far apart in sequence can become neighbors in space:

```text
sequence
Residue 12 -------------------------- Residue 190

3D fold
Residue 12 ---- Residue 190
```

For binding-site prediction, this spatial context can be more informative than sequence position alone. The graph representation makes those physical relationships explicit.

---

## AlphaFold Structure Retrieval

The project uses the **AlphaFold DB REST API** to retrieve structure metadata for a UniProt accession.

For example:

```text
P69905
```

is a UniProt accession identifying a specific protein record.

The retrieval layer:

1. queries the AlphaFold prediction endpoint,
2. selects the available prediction record,
3. extracts the current structure URL,
4. downloads the structure,
5. caches it locally.

This keeps data retrieval separate from the ML model:

```text
REST API → raw structural data
ML pipeline → learn from structural data
```

The code prefers **mmCIF** structures and falls back to PDB when necessary.

---

## Residue Representation

The downloaded structure contains individual atoms. This project simplifies the protein to **one graph node per amino-acid residue**.

Each residue uses its **C-alpha atom** as its representative 3D coordinate:

```text
Residue i
   ↓
C-alpha
   ↓
[x, y, z]
```

For a protein with `N` residues:

```text
pos.shape = [N, 3]
```

---

## Node Features

Each residue is represented using a feature vector containing:

### Amino-acid identity

The 20 standard amino acids are one-hot encoded, with an additional unknown category:

```text
21 dimensions
```

### AlphaFold confidence

AlphaFold provides a per-residue confidence score called **pLDDT**. The model uses:

```text
pLDDT / 100
```

so the value is approximately scaled into the range `0 → 1`.

### Relative sequence position

Each residue also receives its normalized position within the protein:

```text
0.0 ---------------------------- 1.0
start                            end
```

The complete node representation currently contains:

```text
21 amino-acid identity features
+ 1 pLDDT feature
+ 1 normalized sequence-position feature
------------------------------------------------
23 node features
```

Therefore:

```text
x.shape = [N, 23]
```

---

## Building the Protein Graph

Once residue coordinates are extracted, the code calculates pairwise Euclidean distances:

```python
distances = torch.cdist(coords, coords)
```

For residues `i` and `j`:

```text
d(i, j) = sqrt(
    (xi - xj)^2 +
    (yi - yj)^2 +
    (zi - zj)^2
)
```

Residues are connected when their C-alpha atoms are within a chosen radius.

The default cutoff is:

```text
8 Å
```

So:

```text
distance <= 8 Å
        ↓
create graph edge
```

The 8 Å threshold is a **hyperparameter**, not a universal biological constant. It can later be compared against values such as `6 Å`, `10 Å`, or `12 Å` using validation performance.

---

## PyTorch Geometric Representation

Each protein is stored as a PyTorch Geometric `Data` object:

```python
Data(
    x=...,
    pos=...,
    edge_index=...,
    edge_attr=...,
    y=...,
    label_mask=...
)
```

### `x`

Residue feature matrix:

```text
[N, 23]
```

### `pos`

C-alpha coordinates:

```text
[N, 3]
```

### `edge_index`

Graph connectivity:

```text
[2, E]
```

Example:

```text
0 → 4
4 → 0
4 → 9
9 → 4
```

### `edge_attr`

Each edge stores normalized residue-residue distance:

```text
distance / cutoff
```

so the model knows both **which residues are connected** and **how far apart they are**.

### `y`

Binary residue labels:

```text
[N]
```

### `label_mask`

Boolean mask identifying residues with trustworthy labels:

```text
True  = supervised label exists
False = label is unknown
```

This matters because:

```text
missing annotation != confirmed negative
```

Unknown residues are excluded from the loss.

---

## Graph Attention Network

The model uses **Graph Attention Networks (GATs)** through PyTorch Geometric's `GATConv`.

The current architecture is approximately:

```text
23 residue features
        ↓
GAT layer
4 attention heads
        ↓
LayerNorm
ELU
Dropout
        ↓
GAT layer
        ↓
LayerNorm
ELU
        ↓
MLP classifier
        ↓
1 binding-site logit / residue
```

The implementation uses the high-level `GATConv` layer rather than manually implementing attention operations.

---

## What Attention Does

A normal graph neural network lets each node gather information from its neighbors.

A graph-attention network also learns **how important each neighbor is**.

Conceptually:

```text
TYR
 |
 +-- ASP  → high importance
 +-- GLY  → low importance
 +-- LYS  → medium importance
 +-- SER  → low importance
```

The updated representation of `TYR` therefore depends on its own features and a learned weighted combination of its 3D neighbors.

---

## Attention Heads

The first GAT layer uses multiple attention heads.

For example:

```text
hidden_dim = 64
heads = 4
```

Each head has its own learnable attention parameters, allowing the network to represent several kinds of residue relationships at the same time.

The model is not explicitly told what each head should represent. Their behavior is learned through optimization.

With concatenated heads:

```text
64 dimensions/head × 4 heads
=
256-dimensional intermediate representation
```

for each residue after the first attention layer.

---

## Learned Embeddings

Initially, the network receives features that are directly interpretable:

```text
amino-acid identity
pLDDT
sequence position
distance relationships
```

After a GAT layer, each residue becomes a learned vector such as:

```text
[0.21, -0.77, 1.04, ..., 0.35]
```

This is an **embedding**.

Training decides what information should be stored in these dimensions. The embedding can represent combinations of residue identity, local neighborhood composition, distance information, and larger structural context without manually defining every learned feature.

---

## Logits and Probabilities

The final classifier produces one raw number per residue:

```text
logit
```

Example:

```text
Residue 42 →  2.81
Residue 43 → -1.94
```

A logit is converted into a probability using sigmoid:

```python
probabilities = torch.sigmoid(logits)
```

During training, the model passes raw logits directly into:

```python
torch.nn.BCEWithLogitsLoss
```

which combines sigmoid and binary cross-entropy in a numerically stable way.

---

## Training

The core training loop is standard PyTorch:

```python
logits = model(
    batch.x,
    batch.edge_index,
    batch.edge_attr,
)

loss = loss_fn(
    logits[batch.label_mask],
    batch.y[batch.label_mask],
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Each iteration:

```text
1. produces predictions
2. compares predictions with labels
3. computes loss
4. computes gradients
5. updates model parameters
```

Gradients propagate through the classifier and GAT layers, so training changes both the final classification weights and the learned attention behavior.

---

## Class Imbalance

Binding residues are often much rarer than non-binding residues.

For example:

```text
20 binding residues
480 non-binding residues
```

A model predicting everything as non-binding would still achieve 96% accuracy.

To address this, the project computes:

```text
pos_weight = negatives / positives
```

and passes it to:

```python
BCEWithLogitsLoss(pos_weight=...)
```

This makes errors on rare positive residues contribute more strongly to the loss.

---

## Evaluation

The evaluation code reports:

```text
Precision
Recall
F1 score
Average Precision
ROC-AUC
```

### Precision

Of all residues predicted as binding, how many were actually binding?

### Recall

Of all actual binding residues, how many were detected?

### F1

Balances precision and recall.

### Average Precision

Useful for imbalanced binary classification because it summarizes precision-recall behavior across thresholds.

### ROC-AUC

Measures ranking quality between positive and negative classes, but it should not be used as the only metric on strongly imbalanced data.

---

## Labels

AlphaFold provides predicted **structure**, not the ground-truth binding labels required for supervised learning.

Those labels must come from experimental or curated sources.

The intended label-generation pipeline is:

```text
experimental protein-ligand complex
        ↓
identify ligand-contact residues
        ↓
map PDB residue positions to UniProt
        ↓
match positions to AlphaFold residues
        ↓
generate binary labels
```

A residue can be defined as a ligand-contact residue if one or more of its atoms falls within a chosen physical distance of the ligand.

The exact label definition depends on the prediction task. Examples include:

```text
small-molecule ligand binding
ATP binding
metal binding
DNA/RNA binding
protein-protein interfaces
```

These should generally be treated as separate prediction problems unless there is a clear reason to combine them.

---

## Avoiding Data Leakage

Protein datasets require extra care when creating train/validation/test splits.

A random **residue-level** split is invalid because residues from the same protein could appear in both training and testing.

Even splitting by entire protein can leak information if highly similar homologous proteins occur across different splits.

A stronger evaluation setup is:

```text
protein sequences
      ↓
cluster by sequence similarity
      ↓
assign entire clusters to
train / validation / test
```

This makes the held-out test set meaningfully different from the proteins the model learned from.

---

## Dataset Caching

Building one graph requires:

```text
AlphaFold request
↓
structure download
↓
structure parsing
↓
feature construction
↓
distance calculation
↓
graph construction
```

Repeating this every epoch would waste time.

The dataset therefore caches processed PyTorch Geometric graphs:

```text
first request:
AlphaFold → graph construction → save .pt

future requests:
load cached .pt
```

---

## Batching Multiple Proteins

Proteins contain different numbers of residues.

PyTorch Geometric avoids manually padding them to a common length. Its `DataLoader` combines multiple proteins into one disconnected batch graph.

For example:

```text
Protein A → 142 nodes
Protein B → 311 nodes
Protein C → 88 nodes
```

No edges connect separate proteins, so message passing stays inside each protein graph.

---

## Project Structure

```text
protein_binding_gat/
│
├── demo.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── manifest_template.csv
│   └── labels_template.csv
│
└── src/
    ├── __init__.py
    ├── alphafold_io.py
    ├── protein_graph.py
    ├── labels.py
    ├── dataset.py
    ├── model.py
    └── training.py
```

### `src/alphafold_io.py`

Handles:

```text
UniProt accession
↓
AlphaFold REST API
↓
structure URL
↓
local mmCIF/PDB file
```

### `src/protein_graph.py`

Handles structure parsing, C-alpha extraction, residue feature construction, distance computation, graph-edge generation, and PyTorch Geometric `Data` creation.

### `src/labels.py`

Loads residue-level labels and attaches:

```text
graph.y
graph.label_mask
```

### `src/dataset.py`

Implements lazy structure retrieval, graph construction, label attachment, and local graph caching.

### `src/model.py`

Contains the GAT-based residue classifier using:

```text
GATConv
LayerNorm
ELU
Dropout
MLP
```

### `src/training.py`

Contains the training loop, evaluation loop, class-weight calculation, and metrics.

### `demo.py`

Downloads a real AlphaFold structure, builds the graph, and performs a forward pass with an untrained model to verify the pipeline.

### `train_model.py`

Handles argument parsing, datasets, PyG `DataLoader`s, device selection, model initialization, AdamW optimization, checkpointing, validation, and final test evaluation.

---

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

```text
PyTorch
PyTorch Geometric
BioPython
requests
scikit-learn
NumPy
```

---

## Running the Demo

```bash
python demo.py --accession P69905
```

Optional distance cutoff:

```bash
python demo.py \
    --accession P69905 \
    --cutoff 8.0
```

The demo:

```text
downloads the structure
↓
constructs the graph
↓
runs the GAT
↓
prints graph information
↓
prints example residue scores
```

The model is **untrained** in demo mode, so these scores should not be interpreted as biological predictions.

---

## Training

Training requires a manifest and a residue-label CSV:

```bash
python train_model.py \
    --manifest data/manifest.csv \
    --labels data/labels.csv \
    --epochs 50
```

Example configuration:

```bash
python train_model.py \
    --manifest data/manifest.csv \
    --labels data/labels.csv \
    --cutoff 8.0 \
    --hidden-dim 64 \
    --heads 4 \
    --dropout 0.15 \
    --batch-size 4 \
    --epochs 50 \
    --lr 0.001
```

The best checkpoint is selected using validation **average precision**.

---

## Manifest Format

```csv
accession,split
P12345,train
Q67890,train
A12345,val
B67890,test
```

For serious experiments, these splits should be created using protein sequence/family clustering rather than random assignment.

---

## Label Format

```csv
accession,residue_index,label
P12345,1,0
P12345,2,0
P12345,3,1
P12345,4,1
```

The provided label CSV is a **template only** and does not contain a validated biological dataset.

---

## Device Support

The training script checks for:

```text
CUDA
↓
Apple MPS
↓
CPU
```

allowing the same code to run on NVIDIA GPUs, Apple Silicon Macs, or CPU-only systems depending on installed library support.

---

## Current Limitations

### No validated experimental training set yet

The largest remaining research task is constructing and validating residue-level binding labels.

### C-alpha-only structure

Only one coordinate per amino acid is currently used. Side-chain orientation and detailed atom-level chemistry are not explicitly represented.

### Basic node features

The current feature set does not yet include charge, hydrophobicity, solvent accessibility, secondary structure, evolutionary conservation, or protein-language-model embeddings.

### Hard neighborhood cutoff

The graph uses a fixed radius to decide which residues are neighbors, although continuous distance is included as an edge feature.

### Basic GAT architecture

The model provides a clean structural-attention baseline but is not intended to represent the state of the art in geometric protein modeling.

### Homology-safe splitting still required

A meaningful benchmark requires sequence-similarity or family-based clustering before train/validation/test assignment.

---

## Planned Extensions

### 1. Experimental binding labels

Build a pipeline using experimentally resolved protein-ligand complexes:

```text
PDB complexes
↓
ligand contacts
↓
PDB → UniProt residue mapping
↓
AlphaFold graph labels
```

### 2. Baselines

Compare:

```text
MLP
GCN
GAT
```

to test whether 3D graph structure and learned attention actually improve performance.

### 3. Richer residue features

Potential additions:

```text
physicochemical properties
secondary structure
solvent accessibility
evolutionary information
protein-language-model embeddings
```

### 4. Better geometry

Potential extensions include radial-basis distance encodings, orientation features, atom-level graphs, geometric vector perceptrons, and equivariant GNNs.

### 5. Attention analysis

Inspect learned residue-residue attention patterns and compare them with known structural environments.

### 6. Large-scale evaluation

Train and evaluate on a curated, non-redundant protein dataset using leakage-safe sequence clustering.

---

## Research Goal

The project is designed around a measurable question:

> Can residue-level 3D structural context and learned graph attention improve protein binding-site prediction compared with classifiers that only examine individual residue features?

A useful experimental progression is:

```text
Residue-only MLP
        ↓
GCN using 3D neighborhoods
        ↓
GAT using 3D neighborhoods
        ↓
GAT + distance features
        ↓
richer biological features
```

This makes each architectural addition measurable instead of simply making the network more complicated.

---

## Tech Stack

```text
Python
PyTorch
PyTorch Geometric
BioPython
NumPy
scikit-learn
AlphaFold DB REST API
mmCIF / PDB
```

---

## Disclaimer

This repository is currently a project I built to get more familiar with GATs and GNNs, and I also wanted to learn the core of Pytorch and I like BIO so it seemed like a natural fit to build. Unfortunately, its not finished and can't be used yet. 