from __future__ import annotations

from pathlib import Path

import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from torch_geometric.data import Data


AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_AA_INDEX = len(AMINO_ACIDS)
NUM_AA_FEATURES = len(AMINO_ACIDS) + 1

# one-hot amino acid identity + normalized pLDDT + normalized sequence position
NODE_FEATURE_DIM = NUM_AA_FEATURES + 2


def _load_structure(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {path.suffix}")

    return parser.get_structure(path.stem, str(path))


def extract_residue_table(path: str | Path) -> dict[str, object]:
    """
    Convert a PDB/mmCIF structure into one row per standard amino-acid residue.

    We use the C-alpha atom as the residue's representative coordinate.
    AlphaFold stores pLDDT in the B-factor field, so the C-alpha B-factor is
    used as a simple per-residue confidence feature.
    """
    structure = _load_structure(path)

    residue_names: list[str] = []
    chain_ids: list[str] = []
    pdb_residue_numbers: list[int] = []
    insertion_codes: list[str] = []
    coordinates: list[list[float]] = []
    plddt: list[float] = []

    # AlphaFold DB structures normally contain one model. Use the first.
    assert structure is not None 
    model = next(structure.get_models())

    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" not in residue:
                continue

            residue_name = residue.get_resname().upper()
            _, pdb_number, insertion_code = residue.id
            ca = residue["CA"]

            residue_names.append(residue_name)
            chain_ids.append(str(chain.id))
            pdb_residue_numbers.append(int(pdb_number))
            insertion_codes.append(str(insertion_code).strip())
            coordinates.append([float(v) for v in ca.coord])
            assert ca.bfactor is not None
            plddt.append(float(ca.bfactor))

    if not residue_names:
        raise ValueError(f"No standard amino-acid C-alpha atoms found in {path}")

    return {
        "residue_names": residue_names,
        "chain_ids": chain_ids,
        "pdb_residue_numbers": pdb_residue_numbers,
        "insertion_codes": insertion_codes,
        "coords": torch.tensor(coordinates, dtype=torch.float32),
        "plddt": torch.tensor(plddt, dtype=torch.float32),
    }


def make_node_features(
    residue_names: list[str],
    plddt: torch.Tensor,
) -> torch.Tensor:
    """
    Build the feature vector for every residue.

    Features:
      21 dims: amino-acid one-hot (20 standard + unknown)
       1 dim : pLDDT / 100
       1 dim : sequence position scaled to [0, 1]
    """
    n = len(residue_names)
    x = torch.zeros((n, NODE_FEATURE_DIM), dtype=torch.float32)

    for i, name in enumerate(residue_names):
        aa_index = AA_TO_INDEX.get(name, UNKNOWN_AA_INDEX)
        x[i, aa_index] = 1.0

    x[:, NUM_AA_FEATURES] = plddt.clamp(0.0, 100.0) / 100.0

    if n == 1:
        seq_pos = torch.zeros(1, dtype=torch.float32)
    else:
        seq_pos = torch.linspace(0.0, 1.0, steps=n)
    x[:, NUM_AA_FEATURES + 1] = seq_pos

    return x


def build_radius_edges(
    coords: torch.Tensor,
    cutoff_angstrom: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a residue graph from 3D C-alpha distances.

    An edge i -> j is created when the residues are within the cutoff.
    Self edges are excluded here because GATConv can add self-loops itself.

    Returns
    -------
    edge_index : [2, E]
    edge_attr  : [E, 1], containing normalized C-alpha distance
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive")

    distances = torch.cdist(coords, coords)
    n = coords.shape[0]

    mask = distances <= cutoff_angstrom
    mask.fill_diagonal_(False)

    source, target = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([source, target], dim=0).long()

    edge_distances = distances[source, target].unsqueeze(-1)
    edge_attr = edge_distances / cutoff_angstrom

    if edge_index.shape[1] == 0:
        raise ValueError(
            "No graph edges were created. Try a larger distance cutoff."
        )

    return edge_index, edge_attr


def structure_to_graph(
    path: str | Path,
    *,
    accession: str | None = None,
    cutoff_angstrom: float = 8.0,
) -> Data:
    """
    Convert an AlphaFold structure file to a PyTorch Geometric Data object.
    """
    table = extract_residue_table(path)
    coords: torch.Tensor = table["coords"]  # type: ignore[assignment]
    plddt: torch.Tensor = table["plddt"]  # type: ignore[assignment]
    residue_names: list[str] = table["residue_names"]  # type: ignore[assignment]

    x = make_node_features(residue_names, plddt)
    edge_index, edge_attr = build_radius_edges(coords, cutoff_angstrom)

    graph = Data(
        x=x,
        pos=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        residue_index=torch.arange(1, x.shape[0] + 1, dtype=torch.long),
    )

    # Metadata is useful for inspection. Training only depends on tensor fields.
    graph.accession = accession or Path(path).stem
    graph.residue_names = residue_names
    graph.chain_ids = table["chain_ids"]
    graph.pdb_residue_numbers = table["pdb_residue_numbers"]
    graph.insertion_codes = table["insertion_codes"]

    return graph
