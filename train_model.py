from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from src.dataset import ProteinGraphDataset
from src.model import ProteinBindingGAT
from src.protein_graph import NODE_FEATURE_DIM
from src.training import compute_pos_weight, evaluate, train_one_epoch


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    #for mac and stuff
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="best_model.pt")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")

    train_set = ProteinGraphDataset(
        args.manifest,
        args.labels,
        split="train",
        cache_dir=args.cache_dir,
        cutoff_angstrom=args.cutoff,
    )
    val_set = ProteinGraphDataset(
        args.manifest,
        args.labels,
        split="val",
        cache_dir=args.cache_dir,
        cutoff_angstrom=args.cutoff,
    )
    test_set = ProteinGraphDataset(
        args.manifest,
        args.labels,
        split="test",
        cache_dir=args.cache_dir,
        cutoff_angstrom=args.cutoff,
    )

    if len(train_set) == 0:
        raise RuntimeError("Manifest has no train proteins.")
    if len(val_set) == 0:
        raise RuntimeError("Manifest has no val proteins.")
    if len(test_set) == 0:
        raise RuntimeError("Manifest has no test proteins.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    pos_weight_value = compute_pos_weight(train_set)
    print(f"Positive-class weight: {pos_weight_value:.3f}")

    pos_weight = torch.tensor(pos_weight_value, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ProteinBindingGAT(
        input_dim=NODE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_ap = float("-inf")
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val.loss:.4f} "
            f"val_precision={val.precision:.3f} "
            f"val_recall={val.recall:.3f} "
            f"val_f1={val.f1:.3f} "
            f"val_AP={val.average_precision:.3f}"
        )

        if val.average_precision > best_val_ap:
            best_val_ap = val.average_precision
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": NODE_FEATURE_DIM,
                    "hidden_dim": args.hidden_dim,
                    "heads": args.heads,
                    "dropout": args.dropout,
                    "cutoff_angstrom": args.cutoff,
                    "best_val_average_precision": best_val_ap,
                },
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test = evaluate(model, test_loader, loss_fn, device)

    print("\nFinal held-out test metrics")
    print(f"loss:              {test.loss:.4f}")
    print(f"precision:         {test.precision:.4f}")
    print(f"recall:            {test.recall:.4f}")
    print(f"F1:                {test.f1:.4f}")
    print(f"average precision: {test.average_precision:.4f}")
    print(f"ROC-AUC:           {test.roc_auc if test.roc_auc is not None else 'undefined'}")


if __name__ == "__main__":
    main()
