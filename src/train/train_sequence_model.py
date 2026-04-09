"""Train a compact GRU sequence model for next-candle direction."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.policy_utils import (
    build_prediction_frame,
    compute_classification_metrics,
    decision_report,
    sweep_min_edge,
)
from src.features.common import (
    PREDICTION_CONTEXT_COLUMNS,
    build_feature_frame,
    build_prediction_context,
    compute_contiguous_sample_mask,
)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
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
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--feature-set", choices=["baseline", "v2_numeric"], default="baseline")
    p.add_argument("--sweep-min-edge", action="store_true")
    p.add_argument("--sweep-start", type=float, default=0.0)
    p.add_argument("--sweep-end", type=float, default=0.08)
    p.add_argument("--sweep-step", type=float, default=0.01)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def build_sequences(
    df: pd.DataFrame,
    lookback: int,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = build_feature_frame(df, feature_set=feature_set)
    context = build_prediction_context(df, horizon=1)
    y = (df["close"].shift(-1) > df["close"]).astype(int).to_numpy(dtype=np.float32)
    ts = pd.to_datetime(df["timestamp_open"], utc=True)
    arr = features.to_numpy(dtype=np.float32)

    history_bars = max(lookback, 24)
    valid_end_mask = compute_contiguous_sample_mask(
        ts,
        history_bars=history_bars,
        horizon=1,
        timeframe_minutes=5,
    ).to_numpy()

    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    rows: list[dict[str, object]] = []

    for end_idx in range(history_bars - 1, len(df) - 1):
        if not valid_end_mask[end_idx]:
            continue
        seq = arr[end_idx - lookback + 1 : end_idx + 1]
        if np.isnan(seq).any():
            continue

        context_row = context.iloc[end_idx]
        if context_row[PREDICTION_CONTEXT_COLUMNS].isna().any():
            continue

        X_seq.append(seq)
        y_seq.append(y[end_idx])
        rows.append(
            {
                "timestamp_open": pd.to_datetime(context_row["timestamp_open"], utc=True),
                "y": int(y[end_idx]),
                "next_bar_return": float(context_row["next_bar_return"]),
                "realized_vol_24": float(context_row["realized_vol_24"]),
                "volume_ratio_24": float(context_row["volume_ratio_24"]),
                "volume_z_24_context": float(context_row["volume_z_24_context"]),
            }
        )

    meta = pd.DataFrame(rows)
    return np.array(X_seq), np.array(y_seq), meta


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
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(probs)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    X, y, meta = build_sequences(df, args.lookback, feature_set=args.feature_set)

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC")
    ts = pd.to_datetime(meta["timestamp_open"], utc=True)

    train_mask = ts <= train_end
    val_mask = (ts > train_end) & (ts <= val_end)
    test_mask = ts > val_end

    X_train, y_train = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    X_val, y_val = X[val_mask.to_numpy()], y[val_mask.to_numpy()]
    X_test, y_test = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty split after sequence build: train={len(X_train)} val={len(X_val)} test={len(X_test)}"
        )

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(
        input_dim=X.shape[2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
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
        val_metrics = compute_classification_metrics(yv, pv)
        val_auc = float("-inf") if val_metrics["auc"] is None else float(val_metrics["auc"])
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_train_true, train_prob = infer_probs(model, train_loader, device)
    y_val_true, val_prob = infer_probs(model, val_loader, device)
    y_test_true, test_prob = infer_probs(model, test_loader, device)

    train_metrics = compute_classification_metrics(y_train_true, train_prob)
    val_metrics = compute_classification_metrics(y_val_true, val_prob)
    test_metrics = compute_classification_metrics(y_test_true, test_prob)

    prediction_frames = {
        "train": build_prediction_frame(
            meta.loc[train_mask].reset_index(drop=True),
            train_prob,
            split="train",
            model_name="gru",
            feature_set=args.feature_set,
            calibration=None,
        ),
        "val": build_prediction_frame(
            meta.loc[val_mask].reset_index(drop=True),
            val_prob,
            split="val",
            model_name="gru",
            feature_set=args.feature_set,
            calibration=None,
        ),
        "test": build_prediction_frame(
            meta.loc[test_mask].reset_index(drop=True),
            test_prob,
            split="test",
            model_name="gru",
            feature_set=args.feature_set,
            calibration=None,
        ),
    }

    prediction_files: dict[str, str] = {}
    for split_name, pred_df in prediction_frames.items():
        pred_path = out_dir / f"predictions_{split_name}.parquet"
        pred_df.to_parquet(pred_path, index=False)
        prediction_files[split_name] = str(pred_path)

    threshold_sweep = None
    selected_min_edge = float(args.min_edge)
    if args.sweep_min_edge:
        thresholds = np.arange(args.sweep_start, args.sweep_end + (args.sweep_step / 2.0), args.sweep_step)
        threshold_sweep = sweep_min_edge(
            prediction_frames["val"],
            thresholds=thresholds,
            decision_cost=args.decision_cost,
            slippage=args.slippage,
            min_coverage=args.min_coverage,
            auc_tiebreaker=val_metrics["auc"],
        )
        selected_min_edge = float(threshold_sweep["selected_min_edge"])

    report = {
        "model_name": "gru",
        "device": str(device),
        "feature_set": args.feature_set,
        "lookback": args.lookback,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "decision_cost": args.decision_cost,
        "slippage": args.slippage,
        "configured_min_edge": args.min_edge,
        "selected_min_edge": selected_min_edge,
        "threshold_selection": threshold_sweep,
        "samples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "train": {
            **train_metrics,
            "decision": decision_report(
                prediction_frames["train"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "val": {
            **val_metrics,
            "decision": decision_report(
                prediction_frames["val"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "test": {
            **test_metrics,
            "decision": decision_report(
                prediction_frames["test"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "prediction_files": prediction_files,
    }

    torch.save(model.state_dict(), out_dir / "gru_sequence_model.pt")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
