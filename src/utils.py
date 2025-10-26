from __future__ import annotations
import pandas as pd
import numpy as np

def zscore_outlier_clip(df: pd.DataFrame, cols: list[str], z: float = 3.0) -> pd.DataFrame:
    """Clip values outside ±z standard deviations for selected columns."""
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            continue
        lower, upper = mu - z * sd, mu + z * sd
        out[c] = out[c].clip(lower, upper)
    return out

def standardize(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return standardized df and a stats dict for inverse-transform if needed."""
    stats = {}
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0) or 1.0
        stats[c] = (mu, sd)
        out[c] = (out[c] - mu) / sd
    return out, stats

def inverse_standardize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, sd) in stats.items():
        out[c] = out[c] * sd + mu
    return out