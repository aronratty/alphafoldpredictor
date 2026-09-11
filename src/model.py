from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ProteinBindingGAT(nn.Module):
    """
    Residue-level binary classifier.

    GATConv performs graph attention over the edges we created from 3D
    structure. `edge_dim=1` lets the attention mechanism also see the
    normalized residue-residue distance.

    Input:
        x          [N, input_dim]
        edge_index [2, E]
        edge_attr  [E, 1]

    Output:
        logits     [N]
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )

        first_width = hidden_dim * heads
        self.norm1 = nn.LayerNorm(first_width)

        self.gat2 = GATConv(
            in_channels=first_width,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        logits = self.classifier(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x, edge_index, edge_attr))
