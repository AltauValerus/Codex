"""Train a lightweight image-only CNN on rendered candlestick charts.

Colab-friendly defaults (CPU/GPU autodetect).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0


class CandleImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: np.ndarray, transform: transforms.Compose):
        self.image_paths = image_paths
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Root with train/val/test subfolders")
    p.add_argument("--windows", required=True, help="window_index.parquet containing sample_id,y_dir,split")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def collect_split(images_dir: Path, windows: pd.DataFrame, split: str):
    subset = windows[windows["split"] == split][["sample_id", "y_dir"]].copy()
    paths, labels = [], []
    for row in subset.itertuples(index=False):
        p = images_dir / split / f"{row.sample_id}.png"
        if p.exists():
            paths.append(p)
            labels.append(row.y_dir)
    return paths, np.array(labels, dtype=np.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_parquet(args.windows)
    images_dir = Path(args.images_dir)

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    tr_paths, tr_y = collect_split(images_dir, windows, "train")
    va_paths, va_y = collect_split(images_dir, windows, "val")
    te_paths, te_y = collect_split(images_dir, windows, "test")

    train_ds = CandleImageDataset(tr_paths, tr_y, train_tf)
    val_ds = CandleImageDataset(va_paths, va_y, eval_tf)
    test_ds = CandleImageDataset(te_paths, te_y, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)

        train_loss = running / max(len(train_ds), 1)
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_auc={val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    torch.save(model.state_dict(), out_dir / "image_cnn.pt")
    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "samples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "val": val_metrics,
        "test": test_metrics,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
