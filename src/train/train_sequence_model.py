"""Train a compact GRU sequence model for next-candle direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.features.common import build_feature_frame
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    feat["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    feat["ret_12"] = np.log(df["close"] / df["close"].shift(12))
    feat["range_hl"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    feat["body_oc"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    feat["vol_z_24"] = (
        (df["volume"] - df["volume"].rolling(24).mean())
        / (df["volume"].rolling(24).std().replace(0, np.nan))
    )
    return feat


def build_sequences(df: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xf = feature_frame(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    data = Xf.copy()
    data["y"] = y
    data["timestamp_open"] = df["timestamp_open"]
    data = data.dropna().reset_index(drop=True)

    feat_cols = [c for c in data.columns if c not in {"y", "timestamp_open"}]
    arr = data[feat_cols].to_numpy(dtype=np.float32)
    yarr = data["y"].to_numpy(dtype=np.float32)
    ts = data["timestamp_open"].astype("int64").to_numpy()

    X_seq, y_seq, t_seq = [], [], []
    for i in range(lookback - 1, len(data)):
        X_seq.append(arr[i - lookback + 1 : i + 1])
        y_seq.append(yarr[i])
        t_seq.append(ts[i])

    return np.array(X_seq), np.array(y_seq), np.array(t_seq)


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def infer_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    X, y, ts = build_sequences(df, args.lookback)

    train_end = pd.Timestamp(args.train_end, tz="UTC").value
    val_end = pd.Timestamp(args.val_end, tz="UTC").value

    train_mask = ts <= train_end
    val_mask = (ts > train_end) & (ts <= val_end)
    test_mask = ts > val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(input_dim=X.shape[2], hidden_dim=64, num_layers=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)

        train_loss = loss_sum / max(len(train_ds), 1)
        yv, pv = infer_probs(model, val_loader, device)
        m = metrics(yv, pv)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_auc={m['auc']:.4f}")

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    yv, pv = infer_probs(model, val_loader, device)
    yt, pt = infer_probs(model, test_loader, device)
    val_m = metrics(yv, pv)
    test_m = metrics(yt, pt)

    torch.save(model.state_dict(), out_dir / "gru_sequence_model.pt")
    report = {
        "device": str(device),
        "lookback": args.lookback,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "samples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "val": val_m,
        "test": test_m,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
