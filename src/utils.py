"""
utils.py — Shared utilities for the ETA prediction model.

Contains core functions used across all model stages:
- Geospatial calculations (haversine distance)
- Recency-weighted median computation
- Hierarchical Bayesian smoothing for ratio/value tables
- Time parsing and feature derivation helpers
- Guardrail and winsorization utilities
"""

import numpy as np
import pandas as pd
import pytz
from typing import Optional, Tuple, List

# ── Constants ────────────────────────────────────────────────────────────
TZ_KSA = pytz.timezone("Asia/Riyadh")
PEAK_HOURS = set(range(14, 23))  # 14:00–22:00 inclusive

DIST_EDGES  = [0, 1, 2, 4, 6, 8, 12, 20, 1e9]
DIST_LABELS = ["<1", "1-2", "2-4", "4-6", "6-8", "8-12", "12-20", "20+"]

# KSA bounding box for coordinate validation
KSA_LAT_LO, KSA_LAT_HI = 16.0, 33.5
KSA_LON_LO, KSA_LON_HI = 34.0, 57.0


# ── Timestamp Helpers ────────────────────────────────────────────────────
def parse_ksa(s):
    """Parse a timestamp string, localizing to KSA timezone if naive."""
    if pd.isna(s):
        return pd.NaT
    dt = pd.to_datetime(s, errors="coerce")
    if dt is pd.NaT:
        return pd.NaT
    if dt.tzinfo is None:
        try:
            dt = TZ_KSA.localize(dt)
        except Exception:
            return pd.NaT
    return dt


def as_time(s):
    """Safely parse and strip timezone from a timestamp series."""
    x = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert(None)
    except Exception:
        pass
    return x


def coalesce_ts(df, cols):
    """Return the first non-null timestamp from a list of columns."""
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in cols:
        if c in df.columns:
            out = out.where(out.notna(), df[c])
    return out


# ── Geospatial ───────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Compute great-circle distance between two points in kilometres."""
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in vals):
        return float("nan")
    try:
        rlat1, rlon1, rlat2, rlon2 = map(np.radians, map(float, vals))
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return float(6371.0 * c)
    except Exception:
        return float("nan")


def dist_label_for(km: float) -> Optional[str]:
    """Map a distance in km to its bin label."""
    if pd.isna(km):
        return None
    for lo, hi, lab in zip(DIST_EDGES[:-1], DIST_EDGES[1:], DIST_LABELS):
        if lo <= km < hi:
            return lab
    return "20+"


# ── Statistical Utilities ────────────────────────────────────────────────
def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip a Series to [lo, hi] bounds."""
    return s.clip(lower=lo, upper=hi)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of an array."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    cutoff = w.sum() / 2.0
    idx = np.searchsorted(c, cutoff, side="left")
    return float(v[min(idx, len(v) - 1)])


def recency_weights(ts: pd.Series, ref: Optional[pd.Timestamp],
                    half_life_days: float) -> np.ndarray:
    """
    Compute exponential decay weights based on recency.
    
    More recent observations get higher weight; the weight halves
    every `half_life_days` days from the reference timestamp.
    """
    if ref is None or ts is None or ts.isna().all():
        return np.ones(len(ts) if ts is not None else 0, dtype=float)
    dd = (ref - ts).dt.total_seconds() / (3600 * 24)
    lam = np.log(2) / float(half_life_days)
    w = np.exp(-lam * dd.clip(lower=0))
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() == 0:
        w = np.ones(len(w), dtype=float)
    return w


def stacking_bucket(x):
    """Binary flag: 1 if courier is handling stacked (multiple) orders."""
    try:
        return int(float(x) >= 1.0)
    except Exception:
        return 0


# ── Coordinate Quality Flags ────────────────────────────────────────────
T3_MISMATCH_MIN_KM  = 2.0
T3_MISMATCH_RATIO_HI = 3.0
T3_MISMATCH_RATIO_LO = 0.33
T3_OSRM_ZERO_HAV_MINKM = 1.0


def _in_bbox(lat, lon) -> bool:
    try:
        return (KSA_LAT_LO <= float(lat) <= KSA_LAT_HI) and (KSA_LON_LO <= float(lon) <= KSA_LON_HI)
    except Exception:
        return False


def _is_zero_pair(lat, lon) -> bool:
    try:
        return (float(lat) == 0.0 and float(lon) == 0.0)
    except Exception:
        return False


def flag_t3_coord_row(row) -> Tuple[int, str]:
    """
    Validate T3 endpoint coordinates.
    
    Returns (flag, reason) where flag=1 indicates a problematic pair.
    Checks for: zero coordinates, out-of-bounds, missing OSRM, and
    haversine-vs-OSRM distance mismatches.
    """
    plat, plon = row.get("pickup_lat"), row.get("pickup_lon")
    dlat, dlon = row.get("dropoff_lat"), row.get("dropoff_lon")
    osrm_km = row.get("t3_osrm_distance_km")
    hv_km = row.get("t3_dist_haversine_km")

    if _is_zero_pair(plat, plon) or _is_zero_pair(dlat, dlon):
        return 1, "zero_coords"
    if not (_in_bbox(plat, plon) and _in_bbox(dlat, dlon)):
        return 1, "out_of_bbox"
    if pd.isna(osrm_km) or osrm_km < 0:
        return 1, "osrm_dist_missing"

    if pd.notna(hv_km):
        if osrm_km < 0.1 and hv_km >= T3_OSRM_ZERO_HAV_MINKM:
            return 1, "osrm_zero_vs_haversine"
        if max(hv_km, osrm_km) >= T3_MISMATCH_MIN_KM:
            ratio = hv_km / max(osrm_km, 1e-6)
            if ratio >= T3_MISMATCH_RATIO_HI:
                return 1, "hav/osrm_too_high"
            if ratio <= T3_MISMATCH_RATIO_LO:
                return 1, "hav/osrm_too_low"

    return 0, ""


# ── Hierarchical Smoothing ───────────────────────────────────────────────
def gb_apply(gb, func):
    """Compatibility wrapper for pandas groupby.apply across versions."""
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)
