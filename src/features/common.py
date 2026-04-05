"""Canonical feature engineering for BTC next-candle models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
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
