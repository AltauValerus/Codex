"""Shared metrics, prediction export, and decision-policy utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    include_brier: bool = True,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float | None] = {
        "auc": None,
        "f1": None,
        "balanced_accuracy": None,
    }
    if len(y_true) == 0:
        if include_brier:
            metrics["brier"] = None
        return metrics

    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if include_brier:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def build_prediction_frame(
    dataset: pd.DataFrame,
    y_prob: np.ndarray,
    split: str,
    model_name: str,
    feature_set: str,
    calibration: str | None = None,
) -> pd.DataFrame:
    required_cols = {
        "timestamp_open",
        "y",
        "next_bar_return",
        "realized_vol_24",
        "volume_ratio_24",
        "volume_z_24_context",
    }
    missing = required_cols.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing prediction context columns: {sorted(missing)}")

    pred = dataset.loc[
        :,
        [
            "timestamp_open",
            "y",
            "next_bar_return",
            "realized_vol_24",
            "volume_ratio_24",
            "volume_z_24_context",
        ],
    ].copy()
    pred = pred.rename(columns={"y": "y_true"})
    pred["timestamp_open"] = pd.to_datetime(pred["timestamp_open"], utc=True)
    pred["y_prob"] = np.asarray(y_prob, dtype=float)
    pred["split"] = split
    pred["model_name"] = model_name
    pred["feature_set"] = feature_set
    pred["calibration"] = calibration or "none"
    return pred.reset_index(drop=True)


def max_drawdown_pct(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    equity = 100.0 + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def _apply_policy_frame(
    predictions: pd.DataFrame,
    min_edge: float,
    decision_cost: float,
    slippage: float,
) -> pd.DataFrame:
    df = predictions.copy()
    edge = np.abs(df["y_prob"].to_numpy(dtype=float) - 0.5)
    action = np.where(edge >= min_edge, np.where(df["y_prob"].to_numpy(dtype=float) >= 0.5, 1, -1), 0)
    y_signed = np.where(df["y_true"].to_numpy(dtype=int) == 1, 1, -1)

    proxy_pnl = np.zeros(len(df), dtype=float)
    realized_pnl = np.zeros(len(df), dtype=float)
    trade_mask = action != 0
    proxy_pnl[trade_mask] = (
        np.where(action[trade_mask] == y_signed[trade_mask], 1.0, -1.0) - decision_cost - slippage
    )
    realized_pnl[trade_mask] = (
        action[trade_mask] * df.loc[trade_mask, "next_bar_return"].to_numpy(dtype=float) - decision_cost - slippage
    )

    ts = pd.to_datetime(df["timestamp_open"], utc=True)
    df["min_edge"] = min_edge
    df["edge"] = edge
    df["action"] = action
    df["proxy_pnl"] = proxy_pnl
    df["realized_pnl"] = realized_pnl
    df["month"] = ts.dt.strftime("%Y-%m")

    vol_rank = df["realized_vol_24"].rank(pct=True, method="average")
    df["vol_regime"] = np.where(vol_rank < (1.0 / 3.0), "low", np.where(vol_rank < (2.0 / 3.0), "mid", "high"))

    volume_rank = df["volume_ratio_24"].rank(pct=True, method="average")
    df["volume_regime"] = np.where(volume_rank >= 0.5, "high_volume", "low_volume")
    return df


def _slice_summary(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = int(len(df))
    trade_mask = df["action"].to_numpy(dtype=int) != 0
    n_trades = int(np.sum(trade_mask))
    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "n_trades": n_trades,
        "coverage": float(n_trades / max(n_rows, 1)),
        "hit_rate": 0.0,
        "avg_edge": 0.0,
        "proxy_ev_per_trade": 0.0,
        "proxy_total_pnl": 0.0,
        "proxy_max_drawdown_pct": 0.0,
        "realized_ev_per_trade": 0.0,
        "realized_total_pnl": 0.0,
        "realized_max_drawdown_pct": 0.0,
    }
    if n_trades == 0:
        return summary

    traded = df.loc[trade_mask]
    hits = ((traded["y_prob"] >= 0.5).astype(int) == traded["y_true"]).astype(float)
    summary["hit_rate"] = float(hits.mean())
    summary["avg_edge"] = float(traded["edge"].mean())
    summary["proxy_ev_per_trade"] = float(traded["proxy_pnl"].mean())
    summary["proxy_total_pnl"] = float(traded["proxy_pnl"].sum())
    summary["proxy_max_drawdown_pct"] = max_drawdown_pct(traded["proxy_pnl"].to_numpy(dtype=float))
    summary["realized_ev_per_trade"] = float(traded["realized_pnl"].mean())
    summary["realized_total_pnl"] = float(traded["realized_pnl"].sum())
    summary["realized_max_drawdown_pct"] = max_drawdown_pct(traded["realized_pnl"].to_numpy(dtype=float))
    return summary


def _group_summaries(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, sort=True):
        row = {"label": str(key)}
        row.update(_slice_summary(group))
        rows.append(row)
    return rows


def decision_report(
    predictions: pd.DataFrame,
    min_edge: float,
    decision_cost: float,
    slippage: float,
    include_regimes: bool = True,
) -> dict[str, Any]:
    policy_df = _apply_policy_frame(predictions, min_edge=min_edge, decision_cost=decision_cost, slippage=slippage)
    summary = _slice_summary(policy_df)
    summary["min_edge"] = float(min_edge)
    summary["decision_cost"] = float(decision_cost)
    summary["slippage"] = float(slippage)

    if not include_regimes:
        return summary

    summary["regime_slices"] = {
        "monthly": _group_summaries(policy_df, "month"),
        "realized_vol_tercile": _group_summaries(policy_df, "vol_regime"),
        "volume_regime": _group_summaries(policy_df, "volume_regime"),
    }
    return summary


def sweep_min_edge(
    predictions: pd.DataFrame,
    thresholds: list[float] | np.ndarray,
    decision_cost: float,
    slippage: float,
    min_coverage: float = 0.05,
    auc_tiebreaker: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in thresholds:
        row = decision_report(
            predictions,
            min_edge=float(threshold),
            decision_cost=decision_cost,
            slippage=slippage,
            include_regimes=False,
        )
        row["coverage_ok"] = bool(row["coverage"] >= min_coverage)
        row["positive_realized"] = bool(row["realized_total_pnl"] > 0.0)
        rows.append(row)

        auc_key = float("-inf") if auc_tiebreaker is None else float(auc_tiebreaker)
        key = (
            1.0 if row["coverage_ok"] else 0.0,
            1.0 if row["positive_realized"] else 0.0,
            float(row["realized_total_pnl"]),
            auc_key,
            float(row["realized_max_drawdown_pct"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_row = row

    if best_row is None:
        raise ValueError("Threshold sweep requires at least one threshold")

    return {
        "selected_min_edge": float(best_row["min_edge"]),
        "selection_rule": {
            "min_coverage": float(min_coverage),
            "primary": "positive realized_total_pnl after cost/slippage",
            "tie_breaker_1": "higher auc",
            "tie_breaker_2": "higher realized_max_drawdown_pct (less negative)",
        },
        "rows": rows,
    }
