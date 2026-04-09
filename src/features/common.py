"""Canonical feature engineering and dataset helpers for BTC next-candle models."""

from __future__ import annotations

import numpy as np
import pandas as pd


PREDICTION_CONTEXT_COLUMNS = [
    "next_bar_return",
    "realized_vol_24",
    "volume_ratio_24",
    "volume_z_24_context",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _log_return(close: pd.Series, periods: int) -> pd.Series:
    return np.log(close / close.shift(periods))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _cyclical_feature(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = (2.0 * np.pi * values.astype(float)) / period
    return np.sin(angle), np.cos(angle)


def build_feature_frame(df: pd.DataFrame, feature_set: str = "baseline") -> pd.DataFrame:
    if feature_set not in {"baseline", "v2_numeric"}:
        raise ValueError(f"Unsupported feature_set '{feature_set}'")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    timestamps = pd.to_datetime(df["timestamp_open"], utc=True)

    feat = pd.DataFrame(index=df.index)

    # Keep baseline feature definitions exactly backward-compatible.
    feat["ret_1"] = np.log(close / close.shift(1))
    feat["ret_3"] = np.log(close / close.shift(3))
    feat["ret_12"] = np.log(close / close.shift(12))
    feat["range_hl"] = (high - low) / close.replace(0, np.nan)
    feat["body_oc"] = (close - open_) / open_.replace(0, np.nan)
    feat["vol_z_24"] = _rolling_zscore(volume, 24)

    if feature_set == "baseline":
        return feat

    log_ret_1 = feat["ret_1"]
    candle_range = (high - low).replace(0, np.nan)
    sma_12 = close.rolling(12).mean()
    sma_24 = close.rolling(24).mean()
    sma_48 = close.rolling(48).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_24 = close.ewm(span=24, adjust=False).mean()
    ema_48 = close.ewm(span=48, adjust=False).mean()

    for periods in (6, 24, 48):
        feat[f"ret_{periods}"] = _log_return(close, periods)

    for window in (12, 24, 48):
        feat[f"ret_std_{window}"] = log_ret_1.rolling(window).std()
        feat[f"sma_dist_{window}"] = _safe_div(close - close.rolling(window).mean(), close.rolling(window).mean())
        feat[f"ema_dist_{window}"] = _safe_div(close - close.ewm(span=window, adjust=False).mean(), close)
        feat[f"vol_z_{window}"] = _rolling_zscore(volume, window)
        feat[f"vol_ratio_{window}"] = _safe_div(volume, volume.rolling(window).mean())

        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        roll_range = (roll_high - roll_low).replace(0, np.nan)
        feat[f"breakout_high_dist_{window}"] = _safe_div(close - roll_high, close)
        feat[f"breakout_low_dist_{window}"] = _safe_div(close - roll_low, close)
        feat[f"close_pos_range_{window}"] = _safe_div(close - roll_low, roll_range)

    feat["trend_spread_sma_12_48"] = _safe_div(sma_12 - sma_48, close)
    feat["trend_spread_sma_24_48"] = _safe_div(sma_24 - sma_48, close)
    feat["trend_spread_ema_12_48"] = _safe_div(ema_12 - ema_48, close)
    feat["trend_spread_ema_24_48"] = _safe_div(ema_24 - ema_48, close)

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    feat["upper_wick_ratio"] = _safe_div(upper_wick, candle_range)
    feat["lower_wick_ratio"] = _safe_div(lower_wick, candle_range)
    feat["body_to_range"] = _safe_div((close - open_).abs(), candle_range)
    feat["close_pos_in_candle"] = _safe_div(close - low, candle_range)

    hour_sin, hour_cos = _cyclical_feature(timestamps.dt.hour, 24.0)
    dow_sin, dow_cos = _cyclical_feature(timestamps.dt.dayofweek, 7.0)
    feat["hour_sin"] = hour_sin
    feat["hour_cos"] = hour_cos
    feat["dow_sin"] = dow_sin
    feat["dow_cos"] = dow_cos

    return feat


def build_prediction_context(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    log_ret_1 = _log_return(close, 1)

    context = pd.DataFrame(index=df.index)
    context["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    context["next_bar_return"] = close.shift(-horizon) / close - 1.0
    context["realized_vol_24"] = log_ret_1.rolling(24).std()
    context["volume_ratio_24"] = _safe_div(volume, volume.rolling(24).mean())
    context["volume_z_24_context"] = _rolling_zscore(volume, 24)
    return context


def compute_contiguous_sample_mask(
    timestamp_open: pd.Series,
    history_bars: int,
    horizon: int = 1,
    timeframe_minutes: int = 5,
) -> pd.Series:
    """Return a mask for rows whose history and label horizon are gap-free.

    For a sample ending at row ``t`` this requires:
    - the previous ``history_bars`` rows through ``t`` are spaced exactly one timeframe apart
    - the label row at ``t + horizon`` is also exactly one timeframe ahead
    """

    if history_bars < 1:
        raise ValueError("history_bars must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    ts = pd.Series(pd.to_datetime(timestamp_open, utc=True).to_numpy())
    expected = pd.Timedelta(minutes=timeframe_minutes)

    step_ok = ts.diff().eq(expected)
    step_ok.iloc[0] = False

    bad_step_cumsum = (~step_ok).astype(np.int64).cumsum().to_numpy()
    mask = np.zeros(len(ts), dtype=bool)

    valid_end_idx = np.arange(history_bars - 1, len(ts) - horizon)
    start_idx = valid_end_idx - history_bars + 1
    label_idx = valid_end_idx + horizon

    mask[valid_end_idx] = bad_step_cumsum[label_idx] == bad_step_cumsum[start_idx]
    return pd.Series(mask, index=timestamp_open.index)


def build_direction_dataset(
    df: pd.DataFrame,
    history_bars: int = 24,
    horizon: int = 1,
    timeframe_minutes: int = 5,
    feature_set: str = "baseline",
) -> pd.DataFrame:
    """Build a next-candle direction dataset while excluding gap-crossing rows."""

    feat = build_feature_frame(df, feature_set=feature_set)
    y = (df["close"].shift(-horizon) > df["close"]).astype(int)
    timestamps = pd.to_datetime(df["timestamp_open"], utc=True)
    contiguous_mask = compute_contiguous_sample_mask(
        timestamps,
        history_bars=history_bars,
        horizon=horizon,
        timeframe_minutes=timeframe_minutes,
    )

    dataset = feat.copy()
    dataset["y"] = y
    dataset["timestamp_open"] = timestamps
    dataset["is_contiguous"] = contiguous_mask
    dataset = dataset[dataset["is_contiguous"]].drop(columns=["is_contiguous"]).dropna().reset_index(drop=True)
    return dataset
