#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETA Model v5.7.5 — Final Production Model
==========================================
The most refined version of the segmented ETA model, incorporating all
lessons from v5.0–v5.6 iterations.

Major improvements over v5.1:
  - T2 split into T2a (courier travel to vendor via OSRM ratio) and
    T2b (vendor wait/prep time via shop × hour × dow smoothing)
  - Recency-weighted medians with exponential half-life decay (21 days)
    instead of simple medians — adapts to changing delivery patterns
  - Coordinate quality assurance for T3: haversine vs OSRM mismatch
    detection with automatic fallback to zone × hour baseline
  - T2b prep-time priors blended from three tiers: order-level,
    shop-level, and vertical-level historical data
  - T2 bias correction: smoothed zone × hour × stacking residuals
    with minimum activation thresholds (|bias| ≥ 0.5, n ≥ 12)
  - Prior blending via three-source weighted formula:
    blended = (n_cell × cell + n0 × parent + n0_prior × prior) / denom

Outputs per-order predictions, diagnostics by segment, zone × hour
accuracy breakdowns, and coverage reports.
"""

import os, numpy as np, pandas as pd, joblib
from typing import List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

ENR_RECENT = "eta_osrm_jul_enriched.csv"
ORD_PT     = "OrderLevelPTJul.csv"
SHOP_PT    = "ShopLevelPT.csv"
VERT_PT    = "VerticalLevelPT.csv"
T3_CALIB   = "eta_osrm_jul_calibration_t3.csv"  

OUTDIR = "./eta_outputs_v5_7_5"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42

N0_T2A    = 60
N0_T3     = 30
N0_T3_EXT = 80
N0_T2B    = 80
N0_T2BPR  = 120
N0_T2BIAS = 250

HL_T2A   = 21
HL_T3    = 21
HL_T2BPR = 28
HL_BIAS  = 10

MIN_PRED_T2,  MAX_PRED_T2  = 0.5, 240.0
MIN_PRED_T2A, MAX_PRED_T2A = 0.0, 120.0
MIN_PRED_T2B, MAX_PRED_T2B = 0.0,  90.0
MIN_PRED_T3,  MAX_PRED_T3  = 0.5, 180.0
MIN_PRED_T4,  MAX_PRED_T4  = 0.0,  60.0

BIAS_ABS_CAP = 5.0
BIAS_MIN_ABS = 0.5
BIAS_MIN_N   = 12

PEAK_HOURS  = set(range(14,23))
DIST_EDGES  = [0,1,2,4,6,8,12,20,1e9]
DIST_LABELS = ["<1","1-2","2-4","4-6","6-8","8-12","12-20","20+"]

KSA_LAT_LO, KSA_LAT_HI = 16.0, 33.5
KSA_LON_LO, KSA_LON_HI = 34.0, 57.0
T3_MISMATCH_MIN_KM     = 2.0   
T3_MISMATCH_RATIO_HI   = 3.0    
T3_MISMATCH_RATIO_LO   = 0.33  
T3_OSRM_ZERO_HAV_MINKM = 1.0   

def gb_apply(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def dist_label_for(km: float) -> Optional[str]:
    if pd.isna(km): return None
    for lo,hi,lab in zip(DIST_EDGES[:-1], DIST_EDGES[1:], DIST_LABELS):
        if lo <= km < hi: return lab
    return "20+"

def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values)==0: return float("nan")
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    c = np.cumsum(w); cutoff = w.sum()/2.0
    idx = np.searchsorted(c, cutoff, side="left")
    return float(v[min(idx, len(v)-1)])

def recency_weights(ts: pd.Series, ref: Optional[pd.Timestamp], half_life_days: float) -> np.ndarray:
    if ref is None or ts is None or ts.isna().all():
        return np.ones(len(ts) if ts is not None else 0, dtype=float)
    dd = (ref - ts).dt.total_seconds()/(3600*24)
    lam = np.log(2)/float(half_life_days)
    w = np.exp(-lam * dd.clip(lower=0))
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() == 0: w = np.ones(len(w), dtype=float)
    return w

def as_time(s):
    x = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert(None)
    except Exception:
        pass
    return x

def stacking_bucket(x):
    try: return int(float(x) >= 1.0)
    except: return 0

def _in_bbox(lat, lon) -> bool:
    try:
        return (KSA_LAT_LO <= float(lat) <= KSA_LAT_HI) and (KSA_LON_LO <= float(lon) <= KSA_LON_HI)
    except:
        return False

def _is_zero_pair(lat, lon) -> bool:
    try:
        return (float(lat)==0.0 and float(lon)==0.0)
    except:
        return False

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in vals): return float("nan")
    try:
        rlat1, rlon1, rlat2, rlon2 = map(np.radians, map(float, vals))
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat/2.0)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2.0)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return float(6371.0*c)
    except:
        return float("nan")

def flag_t3_coord_row(row) -> Tuple[int, str]:
    """
    Returns (flag, reason) for problematic T3 endpoints.
    """
    plat, plon = row.get("pickup_lat"),  row.get("pickup_lon")
    dlat, dlon = row.get("dropoff_lat"), row.get("dropoff_lon")
    osrm_km    = row.get("t3_osrm_distance_km")
    hv_km      = row.get("t3_dist_haversine_km")

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

enr = pd.read_csv(ENR_RECENT, low_memory=False)
enr.columns = enr.columns.str.strip()

for c in [c for c in enr.columns if c.endswith("_ksa")]:
    enr[c] = as_time(enr[c])

if "order_created_ksa" in enr.columns:
    enr["order_hr_ksa"]  = enr["order_created_ksa"].dt.hour
    enr["order_dow_ksa"] = enr["order_created_ksa"].dt.weekday
else:
    enr["order_hr_ksa"], enr["order_dow_ksa"] = np.nan, np.nan

if ("pickup_hr_ksa" not in enr.columns) or (enr["pickup_hr_ksa"].isna().all()):
    enr["pickup_hr_ksa"] = enr.get("pickup_departure_ksa", pd.NaT).dt.hour

enr["is_peak"] = enr["pickup_hr_ksa"].apply(lambda h: int(h in PEAK_HOURS) if pd.notna(h) else 0)
enr["dow"] = enr["order_dow_ksa"]
if "zone_name_en" not in enr.columns:
    enr["zone_name_en"] = "UNKNOWN"

if "t3_osrm_duration_min" not in enr.columns and "osrm_duration_min" in enr.columns:
    enr = enr.rename(columns={"osrm_duration_min":"t3_osrm_duration_min"})
if "t3_osrm_distance_km" not in enr.columns and "osrm_distance_km" in enr.columns:
    enr = enr.rename(columns={"osrm_distance_km":"t3_osrm_distance_km"})
for c in ["t3_osrm_duration_min","t3_osrm_distance_km","t2_osrm_duration_min","t2_osrm_distance_km"]:
    if c in enr.columns: enr[c] = pd.to_numeric(enr[c], errors="coerce")

end_for_t3 = enr["dropoff_arrived_ksa"].fillna(enr["order_delivered_ksa"])
enr["t1_actual_min"] = (enr["last_accepted_ksa"]    - enr["order_created_ksa"]).dt.total_seconds()/60.0
enr["t2_actual_min"] = (enr["pickup_departure_ksa"] - enr["last_accepted_ksa"]).dt.total_seconds()/60.0
enr["t3_actual_min"] = (end_for_t3                  - enr["pickup_departure_ksa"]).dt.total_seconds()/60.0
enr["ETA_actual_min"]= (end_for_t3                  - enr["order_created_ksa"]).dt.total_seconds()/60.0

enr["t2a_actual_min"] = (enr["pickup_arrived_ksa"]   - enr["last_accepted_ksa"]).dt.total_seconds()/60.0
enr["t2b_actual_min"] = (enr["pickup_departure_ksa"] - enr["pickup_arrived_ksa"]).dt.total_seconds()/60.0
enr["stack_bkt"]      = enr["stacking_order_count"].apply(stacking_bucket)

if {"t2_osrm_duration_min","t2_osrm_distance_km"}.issubset(enr.columns):
    t2a_orders = enr.dropna(subset=["t2a_actual_min","t2_osrm_duration_min","t2_osrm_distance_km","pickup_hr_ksa","zone_name_en"]).copy()
    t2a_orders = t2a_orders[(t2a_orders["t2a_actual_min"].between(0.2, 120.0))
                          & (t2a_orders["t2_osrm_duration_min"].between(0.2, 90.0))]
    t2a_orders["t2_dist_bin"] = t2a_orders["t2_osrm_distance_km"].map(dist_label_for)
    t2a_orders["ratio"] = t2a_orders["t2a_actual_min"] / t2a_orders["t2_osrm_duration_min"]

    ts = t2a_orders["last_accepted_ksa"].fillna(t2a_orders["order_created_ksa"])
    ref = pd.to_datetime(ts, errors="coerce").max()
    t2a_orders["w"] = recency_weights(pd.to_datetime(ts, errors="coerce"), ref, HL_T2A)

    def agg_t2a(g):
        return pd.Series({
            "ratio_p50": weighted_median(g["ratio"].to_numpy(), g["w"].to_numpy()),
            "n": int(len(g))
        })

    t2a_raw = gb_apply(
        t2a_orders.groupby(["zone_name_en","pickup_hr_ksa","t2_dist_bin"], dropna=False),
        agg_t2a
    ).reset_index().rename(columns={"t2_dist_bin":"dist_bin"})
else:
    t2a_raw = pd.DataFrame(columns=["zone_name_en","pickup_hr_ksa","dist_bin","ratio_p50","n"])

def build_smoothed_ratios(df_raw: pd.DataFrame, n0: int,
                          prior: Optional[pd.DataFrame]=None, n0_prior: int=0) -> pd.DataFrame:
    need = ["zone_name_en","pickup_hr_ksa","dist_bin","ratio_p50","n"]
    miss = [c for c in need if c not in df_raw.columns]
    if miss: raise ValueError(f"Missing columns for smoothed ratios: {miss}")
    df = df_raw.copy()
    df["pickup_hr_ksa"] = pd.to_numeric(df["pickup_hr_ksa"], errors="coerce").astype("Int64")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["is_peak"] = df["pickup_hr_ksa"].apply(lambda x: int(x in PEAK_HOURS) if pd.notna(x) else 0)

    hd  = gb_apply(df.groupby(["pickup_hr_ksa","dist_bin"], dropna=False),
        lambda g: pd.Series({"ratio_hd": weighted_median(g["ratio_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_hd": int(g["n"].sum())})
    ).reset_index()
    ipd = gb_apply(df.groupby(["is_peak","dist_bin"], dropna=False),
        lambda g: pd.Series({"ratio_ipd": weighted_median(g["ratio_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_ipd": int(g["n"].sum())})
    ).reset_index()
    h   = gb_apply(df.groupby(["pickup_hr_ksa"], dropna=False),
        lambda g: pd.Series({"ratio_h": weighted_median(g["ratio_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_h": int(g["n"].sum())})
    ).reset_index()
    ip  = gb_apply(df.groupby(["is_peak"], dropna=False),
        lambda g: pd.Series({"ratio_ip": weighted_median(g["ratio_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_ip": int(g["n"].sum())})
    ).reset_index()

    ratio_g = weighted_median(df["ratio_p50"].to_numpy(), df["n"].to_numpy()); n_g = int(df["n"].sum())

    m = (df.merge(hd, on=["pickup_hr_ksa","dist_bin"], how="left")
           .merge(ipd, on=["is_peak","dist_bin"],       how="left")
           .merge(h,  on=["pickup_hr_ksa"],            how="left")
           .merge(ip, on=["is_peak"],                  how="left"))

    def parent(row):
        for rc,nc in [("ratio_hd","n_hd"),("ratio_ipd","n_ipd"),("ratio_h","n_h"),("ratio_ip","n_ip")]:
            r = row.get(rc, np.nan); n = row.get(nc, 0)
            if pd.notna(r) and (n>0): return float(r), int(n), rc
        return float(ratio_g), int(n_g), "ratio_g"
    pr = m.apply(parent, axis=1, result_type="expand")
    m["parent_ratio"], m["parent_n"], m["parent_level"] = pr[0], pr[1], pr[2]

    if prior is not None and len(prior):
        p = prior.copy()
        p = p.rename(columns={"ratio": "ratio_p50", "n_cell":"n"})
        for need in ["zone_name_en","pickup_hr_ksa","dist_bin","ratio_p50","n"]:
            if need not in p.columns:
                if need=="pickup_hr_ksa" and "pickup_hr" in p.columns:
                    p = p.rename(columns={"pickup_hr":"pickup_hr_ksa"})
                else:
                    p[need] = np.nan
        p["pickup_hr_ksa"] = pd.to_numeric(p["pickup_hr_ksa"], errors="coerce").astype("Int64")
        p["n"] = pd.to_numeric(p["n"], errors="coerce").fillna(0).astype(int)
        m = m.merge(
            p.rename(columns={"ratio_p50":"ratio_prior","n":"n_prior"}),
            on=["zone_name_en","pickup_hr_ksa","dist_bin"], how="left"
        )
    else:
        m["ratio_prior"], m["n_prior"] = np.nan, 0

    m["ratio_cell"] = pd.to_numeric(m["ratio_p50"], errors="coerce")
    m["n_cell"] = pd.to_numeric(m["n"], errors="coerce").fillna(0).astype(int)

    num = (m["n_cell"]*m["ratio_cell"].fillna(m["parent_ratio"])
           + n0*m["parent_ratio"]
           + (n0_prior * m["ratio_prior"].fillna(m["parent_ratio"])))
    den = (m["n_cell"] + n0 + (n0_prior * (~m["ratio_prior"].isna()).astype(int)))
    m["ratio_smoothed"] = num / den

    keep = ["zone_name_en","pickup_hr_ksa","dist_bin","is_peak",
            "ratio_cell","n_cell","parent_ratio","parent_n","parent_level",
            "ratio_prior","n_prior","ratio_smoothed"]
    return m[keep].copy()

t2a_smooth = build_smoothed_ratios(t2a_raw, N0_T2A) if len(t2a_raw) else pd.DataFrame(
    {"zone_name_en":["UNKNOWN"],"pickup_hr_ksa":[-1],"dist_bin":["<1"],"is_peak":[0],
     "ratio_cell":[0.0],"n_cell":[0],"parent_ratio":[0.0],"parent_n":[0],
     "parent_level":["none"],"ratio_prior":[np.nan],"n_prior":[0],"ratio_smoothed":[0.0]}
)
t2a_smooth.to_csv(os.path.join(OUTDIR,"t2a_ratio_smoothed.csv"), index=False)

def lookup_ratio(zone: str, hour: int, dist_bin: str, table: pd.DataFrame) -> Tuple[float,str,int]:
    hit = table[(table["zone_name_en"]==zone) & (table["pickup_hr_ksa"]==hour) & (table["dist_bin"]==dist_bin)]
    if len(hit):
        ncell = int(hit["n_cell"].iloc[0])
        return float(hit["ratio_smoothed"].iloc[0]), ("zone_hour_dist_smoothed" if ncell>0 else f"parent:{hit['parent_level'].iloc[0]}"), ncell
    is_peak = int(hour in PEAK_HOURS)
    p = table[(table["is_peak"]==is_peak) & (table["dist_bin"]==dist_bin)]
    if len(p): return float(p["ratio_smoothed"].median()), "fallback:is_peak_dist", 0
    h = table[table["pickup_hr_ksa"]==hour]
    if len(h): return float(h["ratio_smoothed"].median()), "fallback:hour", 0
    return float(table["ratio_smoothed"].median()), "fallback:global", 0

if "t2_osrm_duration_min" in enr.columns:
    enr["t2_dist_bin"] = enr["t2_osrm_distance_km"].map(dist_label_for)
    vals = [lookup_ratio(z, int(h) if pd.notna(h) else -1, b_, t2a_smooth)
            for z,h,b_ in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["t2_dist_bin"])]
    enr["t2a_ratio_p50"] = [v[0] for v in vals]
    enr["t2a_ratio_src"] = [v[1] for v in vals]
    enr["t2a_n_cell"]    = [v[2] for v in vals]
    enr["T2a_P50"] = (enr["t2_osrm_duration_min"] * enr["t2a_ratio_p50"]).clip(MIN_PRED_T2A, MAX_PRED_T2A)
else:
    enr["t2_dist_bin"]=None; enr["t2a_ratio_p50"]=0.0; enr["t2a_ratio_src"]="absent"; enr["t2a_n_cell"]=0
    enr["T2a_P50"]=0.0

if {"pickup_lat","pickup_lon","dropoff_lat","dropoff_lon"}.issubset(enr.columns):
    enr["t3_dist_haversine_km"] = [
        haversine_km(pla, plo, dla, dlo)
        for pla,plo,dla,dlo in zip(enr["pickup_lat"], enr["pickup_lon"], enr["dropoff_lat"], enr["dropoff_lon"])
    ]
else:
    enr["t3_dist_haversine_km"] = np.nan

enr["t3_coord_flag"] = 0
enr["t3_coord_reason"] = ""

tmp_flags = [flag_t3_coord_row(r) for _, r in enr.assign().iterrows()]
if len(tmp_flags):
    enr["t3_coord_flag"]   = [int(f[0]) for f in tmp_flags]
    enr["t3_coord_reason"] = [f[1]      for f in tmp_flags]

if not {"t3_osrm_duration_min","t3_osrm_distance_km"}.issubset(enr.columns):
    raise ValueError("No T3 OSRM columns found (t3_osrm_* or osrm_*).")

t3_orders_base = enr.dropna(subset=["t3_actual_min","t3_osrm_duration_min","t3_osrm_distance_km","pickup_hr_ksa","zone_name_en"]).copy()
t3_orders = t3_orders_base[(t3_orders_base["t3_coord_flag"]==0)]
t3_orders = t3_orders[(t3_orders["t3_actual_min"].between(0.5,180.0))
                    & (t3_orders["t3_osrm_duration_min"].between(0.2,150.0))]
t3_orders["dist_bin"] = t3_orders["t3_osrm_distance_km"].map(dist_label_for)
t3_orders["ratio"]    = t3_orders["t3_actual_min"] / t3_orders["t3_osrm_duration_min"]

ts3 = t3_orders["pickup_departure_ksa"].fillna(t3_orders["order_created_ksa"])
ref3 = pd.to_datetime(ts3, errors="coerce").max()
t3_orders["w"] = recency_weights(pd.to_datetime(ts3, errors="coerce"), ref3, HL_T3)

def agg_t3(g):
    return pd.Series({"ratio_p50": weighted_median(g["ratio"].to_numpy(), g["w"].to_numpy()),
                      "n": int(len(g))})
t3_raw = gb_apply(t3_orders.groupby(["zone_name_en","pickup_hr_ksa","dist_bin"], dropna=False), agg_t3).reset_index()

t3_prior_df = None
if os.path.exists(T3_CALIB):
    ext = pd.read_csv(T3_CALIB, low_memory=False)
    ext.columns = ext.columns.str.strip()
    ext = ext.rename(columns={"ratio":"ratio_p50", "n_cell":"n", "pickup_hr":"pickup_hr_ksa"})
    for need in ["zone_name_en","pickup_hr_ksa","dist_bin","ratio_p50","n"]:
        if need not in ext.columns: ext[need] = np.nan
    ext["pickup_hr_ksa"] = pd.to_numeric(ext["pickup_hr_ksa"], errors="coerce").astype("Int64")
    ext["n"] = pd.to_numeric(ext["n"], errors="coerce").fillna(0).astype(int)
    t3_prior_df = ext[["zone_name_en","pickup_hr_ksa","dist_bin","ratio_p50","n"]].copy()

t3_smooth = build_smoothed_ratios(t3_raw, N0_T3, prior=t3_prior_df, n0_prior=N0_T3_EXT)
t3_smooth.to_csv(os.path.join(OUTDIR,"t3_smoothed_ratios.csv"), index=False)

vals = [lookup_ratio(z, int(h) if pd.notna(h) else -1, b_, t3_smooth)
        for z,h,b_ in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["t3_osrm_distance_km"].map(dist_label_for))]
enr["t3_ratio_p50"]    = [x[0] for x in vals]
enr["t3_ratio_source"] = [x[1] for x in vals]
enr["T3_osrmratio_P50"]= (enr["t3_osrm_duration_min"] * enr["t3_ratio_p50"]).clip(MIN_PRED_T3, MAX_PRED_T3)

def build_t3_baseline(df_good: pd.DataFrame, half_life_days: float, n0: int) -> pd.DataFrame:
    """
    Build smoothed zone×hour baseline for T3 actual minutes (no OSRM dependency).
    Parents: hour, is_peak, global.
    """
    if df_good.empty:
        return pd.DataFrame({"zone_name_en":["UNKNOWN"], "pickup_hr_ksa":[-1], "is_peak":[0],
                             "value_cell":[0.0], "n_cell":[0],
                             "parent_value":[0.0], "parent_n":[0], "parent_level":["none"],
                             "value_smoothed":[0.0]})
    d = df_good.dropna(subset=["t3_actual_min","pickup_hr_ksa","zone_name_en"]).copy()
    d = d[(d["t3_actual_min"].between(0.5, 180.0))]
    d["is_peak"] = d["pickup_hr_ksa"].apply(lambda h: int(h in PEAK_HOURS) if pd.notna(h) else 0)
    ts = d["pickup_departure_ksa"].fillna(d["order_created_ksa"])
    ref = pd.to_datetime(ts, errors="coerce").max()
    d["w"] = recency_weights(pd.to_datetime(ts, errors="coerce"), ref, half_life_days)

    zxh = gb_apply(d.groupby(["zone_name_en","pickup_hr_ksa"], dropna=False),
        lambda g: pd.Series({"value_cell": weighted_median(g["t3_actual_min"].to_numpy(), g["w"].to_numpy()),
                             "n_cell": int(len(g))})
    ).reset_index()

    h = gb_apply(d.groupby(["pickup_hr_ksa"], dropna=False),
        lambda g: pd.Series({"val_h": weighted_median(g["t3_actual_min"].to_numpy(), g["w"].to_numpy()),
                             "n_h": int(len(g))})
    ).reset_index()

    ip = gb_apply(d.groupby(["is_peak"], dropna=False),
        lambda g: pd.Series({"val_ip": weighted_median(g["t3_actual_min"].to_numpy(), g["w"].to_numpy()),
                             "n_ip": int(len(g))})
    ).reset_index()

    val_g = weighted_median(d["t3_actual_min"].to_numpy(), d["w"].to_numpy()); n_g = int(len(d))

    m = (zxh.merge(h, on=["pickup_hr_ksa"], how="left")
            .merge(ip.assign(tmp=1), left_on=zxh["pickup_hr_ksa"].apply(lambda x: 1), right_on="tmp", how="left")
         )
    if "key_0" in m.columns: m = m.drop(columns=["key_0"])
    if "tmp"   in m.columns: m = m.drop(columns=["tmp"])
    m["is_peak"] = m["pickup_hr_ksa"].apply(lambda h: int(h in PEAK_HOURS) if pd.notna(h) else 0)

    def parent(row):
        for vc,nc,lbl in [("val_h","n_h","val_h"),("val_ip","n_ip","val_ip")]:
            v = row.get(vc, np.nan); n = row.get(nc, 0)
            if pd.notna(v) and n>0: return float(v), int(n), lbl
        return float(val_g), int(n_g), "val_g"

    pr = m.apply(parent, axis=1, result_type="expand")
    m["parent_value"], m["parent_n"], m["parent_level"] = pr[0], pr[1], pr[2]
    m["value_smoothed"] = (m["n_cell"]*m["value_cell"] + n0*m["parent_value"]) / (m["n_cell"] + n0)

    keep = ["zone_name_en","pickup_hr_ksa","is_peak",
            "value_cell","n_cell","parent_value","parent_n","parent_level","value_smoothed"]
    return m[keep].copy()

t3_baseline = build_t3_baseline(t3_orders, HL_T3, N0_T3)
t3_baseline.to_csv(os.path.join(OUTDIR,"t3_baseline_zone_hour.csv"), index=False)

def t3_baseline_lookup(zone, hour, is_peak, tbl: pd.DataFrame) -> Tuple[float, str]:
    hit = tbl[(tbl["zone_name_en"]==zone) & (tbl["pickup_hr_ksa"]==hour)]
    if len(hit):
        n = int(hit["n_cell"].iloc[0])
        src = "zone_hour_smoothed" if n>0 else f"parent:{hit['parent_level'].iloc[0]}"
        return float(hit["value_smoothed"].iloc[0]), src
    hh = tbl[tbl["pickup_hr_ksa"]==hour]
    if len(hh): return float(hh["value_smoothed"].median()), "fallback:hour"
    ip = tbl[tbl["is_peak"]==is_peak]
    if len(ip): return float(ip["value_smoothed"].median()), "fallback:is_peak"
    return float(tbl["value_smoothed"].median()), "fallback:global"

t3_mode = []
t3_base_vals = []
t3_base_srcs = []
for z,h,pk,flag,osrm_pred in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["is_peak"],
                                 enr["t3_coord_flag"], enr["T3_osrmratio_P50"]):
    if int(flag)==1:
        v, src = t3_baseline_lookup(z, int(h) if pd.notna(h) else -1, int(pk) if pd.notna(pk) else 0, t3_baseline)
        t3_mode.append("zone_hour_fallback")
        t3_base_vals.append(np.clip(v, MIN_PRED_T3, MAX_PRED_T3))
        t3_base_srcs.append(src)
    else:
        t3_mode.append("osrm_ratio")
        t3_base_vals.append(osrm_pred)
        t3_base_srcs.append("osrm_ratio")

enr["t3_pred_mode"] = t3_mode
enr["T3_base_P50"]  = pd.to_numeric(t3_base_vals, errors="coerce").clip(MIN_PRED_T3, MAX_PRED_T3)
enr["t3_ratio_source_fallback"] = t3_base_srcs

have_arrived = enr["dropoff_arrived_ksa"].notna()
have_deliv   = enr["order_delivered_ksa"].notna()
mask_dw_train= have_arrived & have_deliv
enr["dwell_min"] = (enr["order_delivered_ksa"] - enr["dropoff_arrived_ksa"]).dt.total_seconds()/60.0
enr.loc[mask_dw_train, "dwell_min"] = enr.loc[mask_dw_train, "dwell_min"].clip(0,60)

dw = enr.loc[mask_dw_train, ["zone_name_en","pickup_hr_ksa","dwell_min"]].assign(n=1)
if len(dw):
    dw_hour = (dw.groupby("pickup_hr_ksa")
                 .agg(dwell_h=("dwell_min","median"), n_h=("n","sum"))
                 .reset_index())
    dwell_g = float(dw["dwell_min"].median())
    dw_m = dw.merge(dw_hour, on="pickup_hr_ksa", how="left")
    zxh = (dw_m.groupby(["zone_name_en","pickup_hr_ksa"])
             .agg(dwell_cell=("dwell_min","median"),
                  n_cell=("n","sum"),
                  dwell_parent=("dwell_h","median"),
                  n_parent=("n_h","median"))
             .reset_index())
    zxh["dwell_smoothed"] = (zxh["n_cell"]*zxh["dwell_cell"] + N0_T2B*zxh["dwell_parent"]) / (zxh["n_cell"] + N0_T2B)
else:
    dw_hour = pd.DataFrame({"pickup_hr_ksa":[-1],"dwell_h":[0.0],"n_h":[0]})
    dwell_g = 0.0
    zxh = pd.DataFrame({"zone_name_en":["UNKNOWN"],"pickup_hr_ksa":[-1],"dwell_cell":[0.0],
                        "n_cell":[0],"dwell_parent":[0.0],"n_parent":[0],"dwell_smoothed":[0.0]})
zxh.to_csv(os.path.join(OUTDIR,"dwell_smoothed.csv"), index=False)

def dwell_lookup(zone, hour):
    hit = zxh[(zxh["zone_name_en"]==zone) & (zxh["pickup_hr_ksa"]==hour)]
    if len(hit): return float(hit["dwell_smoothed"].iloc[0])
    hh = dw_hour[dw_hour["pickup_hr_ksa"]==hour]
    if len(hh): return float(hh["dwell_h"].iloc[0])
    return float(dwell_g)

needs_dwell = enr["dropoff_arrived_ksa"].isna() & enr["order_delivered_ksa"].notna()
enr["T4_dwell_P50"] = [0.0 if not nd else float(np.clip(dwell_lookup(z, int(h) if pd.notna(h) else -1), MIN_PRED_T4, MAX_PRED_T4))
                        for nd,z,h in zip(needs_dwell, enr["zone_name_en"], enr["pickup_hr_ksa"])]
enr["T3_P50"] = enr["T3_base_P50"] + enr["T4_dwell_P50"]

def prior_from_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["shop_id","sub_vertical_name","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d = df.copy()
    if "shop_id" not in d.columns and "mbo_chain_id" in d.columns: d = d.rename(columns={"mbo_chain_id":"shop_id"})
    if "sub_vertical_name" not in d.columns and "sub_vertical_detail" in d.columns:
        d = d.rename(columns={"sub_vertical_detail":"sub_vertical_name"})
    if "prep_time_min" not in d.columns:
        if {"accepted_at","picked_up_at"}.issubset(d.columns):
            d["prep_time_min"] = (as_time(d["picked_up_at"]) - as_time(d["accepted_at"])).dt.total_seconds()/60.0
        else:
            return pd.DataFrame(columns=["shop_id","sub_vertical_name","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d["ts"] = as_time(d.get("picked_up_at", d.get("accepted_at")))
    d["pickup_hr_ksa"] = as_time(d.get("accepted_at")).dt.hour
    d["dow"] = as_time(d.get("accepted_at")).dt.weekday
    d = d.dropna(subset=["prep_time_min"])
    return d[["shop_id","sub_vertical_name","pickup_hr_ksa","dow","prep_time_min","ts"]].assign(n=1)

def prior_from_shop(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["shop_id","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d = df.copy()
    if "shop_id" not in d.columns and "mbo_chain_id" in d.columns: d = d.rename(columns={"mbo_chain_id":"shop_id"})
    if "prep_time_min" not in d.columns:
        if "prep_time_sec" in d.columns:
            d["prep_time_min"] = pd.to_numeric(d["prep_time_sec"], errors="coerce")/60.0
        else:
            return pd.DataFrame(columns=["shop_id","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d["ts"] = as_time(d.get("picked_up_at", d.get("accepted_at")))
    if "pickup_hr_ksa" not in d.columns: d["pickup_hr_ksa"] = -1
    if "dow" not in d.columns: d["dow"] = -1
    return d[["shop_id","pickup_hr_ksa","dow","prep_time_min","ts"]].assign(n=1)

def prior_from_vertical(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["sub_vertical_name","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d = df.copy()
    if "sub_vertical_name" not in d.columns and "sub_vertical_detail" in d.columns:
        d = d.rename(columns={"sub_vertical_detail":"sub_vertical_name"})
    if "prep_time_min" not in d.columns:
        if "prep_time_sec" in d.columns:
            d["prep_time_min"] = pd.to_numeric(d["prep_time_sec"], errors="coerce")/60.0
        else:
            return pd.DataFrame(columns=["sub_vertical_name","pickup_hr_ksa","dow","prior_p50","n_prior","ts"])
    d["ts"] = as_time(d.get("picked_up_at", d.get("accepted_at")))
    if "pickup_hr_ksa" not in d.columns: d["pickup_hr_ksa"] = -1
    if "dow" not in d.columns: d["dow"] = -1
    return d[["sub_vertical_name","pickup_hr_ksa","dow","prep_time_min","ts"]].assign(n=1)

def build_prior(df: pd.DataFrame, keys: List[str], half_life_days: float) -> pd.DataFrame:
    if df.empty: 
        cols = keys + ["pickup_hr_ksa","dow","prior_p50","n_prior"]
        return pd.DataFrame(columns=cols)
    ref = as_time(df["ts"]).max()
    df["w"] = recency_weights(as_time(df["ts"]), ref, half_life_days)
    def agg(g):
        return pd.Series({
            "prior_p50": weighted_median(g["prep_time_min"].to_numpy(), g["w"].to_numpy()),
            "n_prior": int(len(g))
        })
    return gb_apply(df.groupby(keys+["pickup_hr_ksa","dow"], dropna=False), agg).reset_index()

ordpt = pd.read_csv(ORD_PT,  low_memory=False) if os.path.exists(ORD_PT)  else pd.DataFrame()
shoppt= pd.read_csv(SHOP_PT, low_memory=False) if os.path.exists(SHOP_PT) else pd.DataFrame()
vertpt= pd.read_csv(VERT_PT, low_memory=False) if os.path.exists(VERT_PT) else pd.DataFrame()

po = prior_from_order(ordpt); ps = prior_from_shop(shoppt); pv = prior_from_vertical(vertpt)
pri_order = build_prior(po, ["shop_id","sub_vertical_name"], HL_T2BPR)
pri_shop  = build_prior(ps, ["shop_id"], HL_T2BPR)
pri_vert  = build_prior(pv, ["sub_vertical_name"], HL_T2BPR)

t2b_tr = enr.dropna(subset=["t2b_actual_min","pickup_hr_ksa"]).copy()
if "shop_id" not in t2b_tr.columns: t2b_tr["shop_id"] = -1
if "sub_vertical_name" not in t2b_tr.columns: t2b_tr["sub_vertical_name"] = "UNKNOWN"
t2b_tr = t2b_tr[(t2b_tr["t2b_actual_min"].between(0.0, 90.0))]
t2b_tr["dow"] = t2b_tr["order_dow_ksa"]

t2b_raw = (t2b_tr.groupby(["shop_id","sub_vertical_name","pickup_hr_ksa","dow"], dropna=False)
           .agg(value_p50=("t2b_actual_min","median"),
                n=("t2b_actual_min","size")).reset_index())

def build_smoothed_values(df_raw: pd.DataFrame, n0: int) -> pd.DataFrame:
    need = ["shop_id","sub_vertical_name","pickup_hr_ksa","dow","value_p50","n"]
    miss = [c for c in need if c not in df_raw.columns]
    if miss: raise ValueError(f"Missing columns for smoothed values: {miss}")
    d = df_raw.copy()
    d["pickup_hr_ksa"] = pd.to_numeric(d["pickup_hr_ksa"], errors="coerce").astype("Int64")
    d["dow"] = pd.to_numeric(d["dow"], errors="coerce").astype("Int64")
    d["n"] = pd.to_numeric(d["n"], errors="coerce").fillna(0).astype(int)

    svhd = gb_apply(d.groupby(["sub_vertical_name","pickup_hr_ksa","dow"], dropna=False),
        lambda g: pd.Series({"val_svh": weighted_median(g["value_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_svh": int(g["n"].sum())})
    ).reset_index()
    hd   = gb_apply(d.groupby(["pickup_hr_ksa","dow"], dropna=False),
        lambda g: pd.Series({"val_hd": weighted_median(g["value_p50"].to_numpy(), g["n"].to_numpy()),
                             "n_hd": int(g["n"].sum())})
    ).reset_index()
    val_g = weighted_median(d["value_p50"].to_numpy(), d["n"].to_numpy()); n_g = int(d["n"].sum())

    base = d[["shop_id","sub_vertical_name","pickup_hr_ksa","dow","value_p50","n"]].copy()
    m = (base.merge(svhd, on=["sub_vertical_name","pickup_hr_ksa","dow"], how="left")
             .merge(hd,   on=["pickup_hr_ksa","dow"],                 how="left"))

    def parent(row):
        for vc,nc in [("val_svh","n_svh"),("val_hd","n_hd")]:
            v = row.get(vc, np.nan); n = row.get(nc, 0)
            if pd.notna(v) and (n>0): return float(v), int(n), vc
        return float(val_g), int(n_g), "val_g"

    pr = m.apply(parent, axis=1, result_type="expand")
    m["parent_value"], m["parent_n"], m["parent_level"] = pr[0], pr[1], pr[2]
    m["value_cell"] = pd.to_numeric(m["value_p50"], errors="coerce")
    m["n_cell"] = m["n"].astype(int)
    m["value_smoothed"] = (m["n_cell"]*m["value_cell"].fillna(m["parent_value"]) + n0*m["parent_value"]) / (m["n_cell"] + n0)
    keep = ["shop_id","sub_vertical_name","pickup_hr_ksa","dow",
            "value_cell","n_cell","parent_value","parent_n","parent_level","value_smoothed"]
    return m[keep].copy()

t2b_smooth_in = build_smoothed_values(t2b_raw, N0_T2B)

def left_merge_prior(base: pd.DataFrame, pri: pd.DataFrame, keys: List[str],
                     hour_col="pickup_hr_ksa", dow_col="dow") -> pd.DataFrame:
    """
    Safe left-merge that fills 'prior_p50' & 'n_prior' from priors (supports any-hour/-dow rows).
    Avoids fillna-with-Series and column collisions by renaming RHS columns and using combine_first.
    """
    out = base.copy()
    if "prior_p50" not in out.columns: out["prior_p50"] = np.nan
    if "n_prior"   not in out.columns: out["n_prior"]   = np.nan

    if pri.empty:
        out["n_prior"] = out["n_prior"].fillna(0).astype(int)
        return out

    def merge_once(df_left: pd.DataFrame, df_right: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
        if not all(c in df_right.columns for c in on_cols): 
            return df_left
        r = df_right[on_cols + ["prior_p50","n_prior"]].copy()
        r = r.rename(columns={"prior_p50":"_prior_p50_src", "n_prior":"_n_prior_src"})
        m = df_left.merge(r, on=on_cols, how="left")
        if "_prior_p50_src" in m.columns:
            m["prior_p50"] = m["prior_p50"].combine_first(m["_prior_p50_src"])
            m = m.drop(columns=["_prior_p50_src"])
        if "_n_prior_src" in m.columns:
            m["n_prior"] = m["n_prior"].combine_first(m["_n_prior_src"])
            m = m.drop(columns=["_n_prior_src"])
        return m

    out = merge_once(out, pri, keys + [hour_col, dow_col])

    if (hour_col in pri.columns) and ((pri[hour_col]==-1).any()):
        pri_any_h = pri[pri[hour_col]==-1].drop(columns=[hour_col]).copy()
        out = merge_once(out, pri_any_h, keys + [dow_col])

    if (dow_col in pri.columns) and ((pri[dow_col]==-1).any()):
        pri_any_d = pri[pri[dow_col]==-1].drop(columns=[dow_col]).copy()
        out = merge_once(out, pri_any_d, keys + [hour_col])

    out["n_prior"]   = pd.to_numeric(out["n_prior"], errors="coerce").fillna(0).astype(int)
    out["prior_p50"] = pd.to_numeric(out["prior_p50"], errors="coerce")
    return out

b = t2b_smooth_in.copy()

tmp = left_merge_prior(b, pri_order, keys=["shop_id","sub_vertical_name"])
b[["prior_p50","n_prior"]] = tmp[["prior_p50","n_prior"]].to_numpy()

miss = b["prior_p50"].isna()
if miss.any():
    tmp = left_merge_prior(b.loc[miss].copy(), pri_shop, keys=["shop_id"])
    b.loc[miss, ["prior_p50","n_prior"]] = tmp[["prior_p50","n_prior"]].to_numpy()

miss = b["prior_p50"].isna()
if miss.any():
    tmp = left_merge_prior(b.loc[miss].copy(), pri_vert, keys=["sub_vertical_name"])
    b.loc[miss, ["prior_p50","n_prior"]] = tmp[["prior_p50","n_prior"]].to_numpy()

b["prior_p50"] = b["prior_p50"].fillna(b["parent_value"])
b["n_prior"]   = b["n_prior"].fillna(0).astype(int)

num = (b["n_cell"]*b["value_cell"].fillna(b["parent_value"])
       + N0_T2B*b["parent_value"]
       + N0_T2BPR*b["prior_p50"])
den = (b["n_cell"] + N0_T2B + N0_T2BPR)
b["value_blended"] = num / den

t2b_blend = b.drop(columns=[c for c in b.columns if c == "value_smoothed"]).rename(
    columns={"value_blended":"value_smoothed"}
).copy()
t2b_blend = t2b_blend.loc[:, ~t2b_blend.columns.duplicated()].copy()
t2b_blend.to_csv(os.path.join(OUTDIR,"t2b_smoothed.csv"), index=False)
b[["shop_id","sub_vertical_name","pickup_hr_ksa","dow","n_cell","parent_value","prior_p50","n_prior","value_blended"]].to_csv(
    os.path.join(OUTDIR,"t2b_prior_blend_summary.csv"), index=False
)

def train_t1(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    cat = [c for c in ["sub_vertical_name","zone_name_en","business_branch_name","shop_id","mbo_chain_id"] if c in df.columns]
    num = [c for c in ["stacking_order_count","accepted_courier_count","rejected_courier_count",
                       "pickup_lat","pickup_lon","dropoff_lat","dropoff_lon",
                       "order_hr_ksa","order_dow_ksa"] if c in df.columns]
    if df.empty or df["t1_actual_min"].notna().sum() < 200:
        class Med:
            def __init__(self, m): self.m=m
            def predict(self, X): return np.full(len(X), self.m, dtype=float)
        median_t1 = float(df["t1_actual_min"].median()) if df["t1_actual_min"].notna().any() else 5.0
        return Med(median_t1), [], []
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
    ])
    model = Pipeline([("prep", pre),
                      ("gb", GradientBoostingRegressor(loss="quantile", alpha=0.5,
                                                       max_depth=4, n_estimators=600,
                                                       min_samples_leaf=20, random_state=SEED))])
    model.fit(df[cat+num], df["t1_actual_min"])
    return model, cat, num

t1_df = enr.dropna(subset=["t1_actual_min"]).query("0 <= t1_actual_min <= 60").copy()
t1_model, t1_cat, t1_num = train_t1(t1_df)
joblib.dump(t1_model, os.path.join(OUTDIR,"t1_p50.joblib"))
enr["T1_P50"] = t1_model.predict(
    enr.reindex(columns=t1_cat+t1_num, fill_value=np.nan) if (t1_cat or t1_num)
    else enr.assign(dummy=1)[["dummy"]]
)

if "shop_id" not in enr.columns: enr["shop_id"] = -1
if "sub_vertical_name" not in enr.columns: enr["sub_vertical_name"] = "UNKNOWN"
enr["dow"] = enr["order_dow_ksa"]

def value_lookup(shop_id, sub_vertical, hour, dow, table: pd.DataFrame):
    hit = table[(table["shop_id"]==shop_id) & (table["pickup_hr_ksa"]==hour) & (table["dow"]==dow)]
    if len(hit): 
        return float(hit["value_smoothed"].iloc[0]), "shop_hour_dow", int(hit["n_cell"].iloc[0])
    sv = table[(table["sub_vertical_name"]==sub_vertical) & (table["pickup_hr_ksa"]==hour) & (table["dow"]==dow)]
    if len(sv): return float(sv["value_smoothed"].median()), "fallback:subvertical_hour_dow", 0
    hd = table[(table["pickup_hr_ksa"]==hour) & (table["dow"]==dow)]
    if len(hd): return float(hd["value_smoothed"].median()), "fallback:hour_dow", 0
    return float(table["value_smoothed"].median()), "fallback:global", 0

vals = [value_lookup(s, sv, int(h) if pd.notna(h) else -1, int(d) if pd.notna(d) else -1, t2b_blend)
        for s,sv,h,d in zip(enr["shop_id"], enr["sub_vertical_name"], enr["pickup_hr_ksa"], enr["dow"])]
enr["t2b_value_p50"] = [v[0] for v in vals]
enr["t2b_value_src"] = [v[1] for v in vals]
enr["t2b_n_cell"]    = [v[2] for v in vals]
enr["T2b_base_P50"]  = pd.to_numeric(enr["t2b_value_p50"], errors="coerce").clip(MIN_PRED_T2B, MAX_PRED_T2B)

enr["T2b_resid_P50"] = 0.0
t2b_train = enr.dropna(subset=["t2b_actual_min","T2b_base_P50"]).copy()
if len(t2b_train) >= 1000:
    t2b_train["t2b_resid"] = winsorize(t2b_train["t2b_actual_min"] - t2b_train["T2b_base_P50"], -10.0, 30.0)
    t2b_train = t2b_train.sort_values("order_created_ksa")
    cut_idx = int(0.8 * len(t2b_train))
    t2b_trn = t2b_train.iloc[:cut_idx].copy()

    t2b_cat = [c for c in ["zone_name_en","sub_vertical_name","business_branch_name","shop_id","mbo_chain_id"] if c in t2b_trn.columns]
    t2b_num = [c for c in ["pickup_hr_ksa","dow","is_peak","stack_bkt",
                           "stacking_order_count","accepted_courier_count","rejected_courier_count",
                           "t2_osrm_distance_km","t2_osrm_duration_min",
                           "t2a_actual_min","t2a_ratio_p50","T2a_P50"] if c in t2b_trn.columns]

    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=15))]), t2b_cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), t2b_num),
    ])
    t2b_resid_model = Pipeline([
        ("prep", pre),
        ("gb", GradientBoostingRegressor(
            loss="quantile", alpha=0.5,
            max_depth=4, n_estimators=500, min_samples_leaf=25, random_state=SEED
        ))
    ])
    t2b_resid_model.fit(t2b_trn[t2b_cat+t2b_num], t2b_trn["t2b_resid"])
    joblib.dump(t2b_resid_model, os.path.join(OUTDIR,"t2b_resid_p50.joblib"))
    enr["T2b_resid_P50"] = t2b_resid_model.predict(enr.reindex(columns=t2b_cat+t2b_num, fill_value=np.nan))

def gate_resid(resid):
    if pd.isna(resid): return 0.0
    if abs(resid) <= 0.5: return float(resid)
    if resid > 0:  return float(min(resid, 10.0))
    else:          return float(max(resid, -6.0))

enr["T2b_resid_gated"] = enr["T2b_resid_P50"].apply(gate_resid)
enr["T2b_P50"] = (enr["T2b_base_P50"] + enr["T2b_resid_gated"]).clip(MIN_PRED_T2B, MAX_PRED_T2B)

enr["T2a_P50"] = pd.to_numeric(enr["T2a_P50"], errors="coerce").fillna(0.0)
enr["T2_P50"]  = (enr["T2a_P50"] + enr["T2b_P50"]).clip(MIN_PRED_T2, MAX_PRED_T2)

enr_sorted = enr.sort_values("order_created_ksa")
cut_t2 = int(0.8 * len(enr_sorted))
train_bias = enr_sorted.iloc[:cut_t2].copy()
train_bias["t2_resid"] = train_bias["t2_actual_min"] - (train_bias["T2a_P50"].fillna(0) + train_bias["T2b_P50"].fillna(0))
train_bias = train_bias.dropna(subset=["t2_resid","pickup_hr_ksa","zone_name_en"])
train_bias["stack_bkt"] = train_bias["stack_bkt"].fillna(0).astype(int)
train_bias["is_peak"]   = train_bias["pickup_hr_ksa"].apply(lambda h: int(h in PEAK_HOURS) if pd.notna(h) else 0)
ref_ts = as_time(train_bias["order_created_ksa"]).max()
train_bias["w"] = recency_weights(as_time(train_bias["order_created_ksa"]), ref_ts, HL_BIAS)

g_zhb = (train_bias
         .groupby(["zone_name_en","pickup_hr_ksa","stack_bkt"], dropna=False)
         .apply(lambda g: pd.Series({
             "bias_cell": weighted_median(g["t2_resid"].to_numpy(), g["w"].to_numpy()),
             "n_cell": int(len(g))
         }))
         .reset_index())
bh  = gb_apply(train_bias.groupby(["zone_name_en","pickup_hr_ksa"], dropna=False),
               lambda g: pd.Series({"bias_zh": weighted_median(g["t2_resid"].to_numpy(), g["w"].to_numpy()),
                                    "n_zh": int(len(g))})).reset_index()
hb  = gb_apply(train_bias.groupby(["pickup_hr_ksa","stack_bkt"], dropna=False),
               lambda g: pd.Series({"bias_hb": weighted_median(g["t2_resid"].to_numpy(), g["w"].to_numpy()),
                                    "n_hb": int(len(g))})).reset_index()
h   = gb_apply(train_bias.groupby(["pickup_hr_ksa"], dropna=False),
               lambda g: pd.Series({"bias_h": weighted_median(g["t2_resid"].to_numpy(), g["w"].to_numpy()),
                                    "n_h": int(len(g))})).reset_index()
bias_g = weighted_median(train_bias["t2_resid"].to_numpy(), train_bias["w"].to_numpy())

m = (g_zhb.merge(bh, on=["zone_name_en","pickup_hr_ksa"], how="left")
          .merge(hb, on=["pickup_hr_ksa","stack_bkt"],    how="left")
          .merge(h,  on=["pickup_hr_ksa"],                how="left"))

def pick_parent(row):
    for vc,nc in [("bias_zh","n_zh"), ("bias_hb","n_hb"), ("bias_h","n_h")]:
        v = row.get(vc, np.nan); n = row.get(nc, 0)
        if pd.notna(v) and n>0: return float(v), int(n), vc
    return float(bias_g), int(len(train_bias)), "bias_g"

pr = m.apply(pick_parent, axis=1, result_type="expand")
m["parent_bias"], m["parent_n"], m["parent_level"] = pr[0], pr[1], pr[2]
m["bias_smoothed"] = (m["n_cell"]*m["bias_cell"] + N0_T2BIAS*m["parent_bias"]) / (m["n_cell"] + N0_T2BIAS)

t2_bias_map = m[["zone_name_en","pickup_hr_ksa","stack_bkt","bias_cell","n_cell","parent_bias","parent_level","bias_smoothed"]].copy()
t2_bias_map.to_csv(os.path.join(OUTDIR,"t2_bias_map_stk_smoothed.csv"), index=False)

def bias_lookup(zone, hour, stk):
    hit = t2_bias_map[(t2_bias_map["zone_name_en"]==zone) & (t2_bias_map["pickup_hr_ksa"]==hour) & (t2_bias_map["stack_bkt"]==stk)]
    if len(hit):
        b = float(hit["bias_smoothed"].iloc[0]); n = int(hit["n_cell"].iloc[0])
        if n >= BIAS_MIN_N and abs(b) >= BIAS_MIN_ABS:
            return float(np.clip(b, -BIAS_ABS_CAP, BIAS_ABS_CAP))
    return 0.0

enr["T2_bias_corr"]   = [bias_lookup(z, int(h) if pd.notna(h) else -1, int(stk) if pd.notna(stk) else 0)
                         for z,h,stk in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["stack_bkt"])]
enr["T2_P50_CORR"]    = (enr["T2_P50"] + enr["T2_bias_corr"]).clip(MIN_PRED_T2, MAX_PRED_T2)

enr["ETA_P50"]      = enr["T1_P50"] + enr["T2_P50"]      + enr["T3_P50"]
enr["ETA_P50_CORR"] = enr["T1_P50"] + enr["T2_P50_CORR"] + enr["T3_P50"]

def metrics(actual: pd.Series, pred: pd.Series, mask: pd.Series):
    e = (actual - pred).abs()
    e = e[mask]
    if e.empty: return (np.nan, np.nan, 0)
    return (float(e.mean()), float(e.median()), int(mask.sum()))

mask_T1  = enr["t1_actual_min"].between(0,60)
mask_T2  = enr["t2_actual_min"].between(0.5,240)
mask_T3  = enr["t3_actual_min"].between(0.5,180)
mask_ETA = enr["ETA_actual_min"].between(10,300)

m_T1  = metrics(enr["t1_actual_min"],  enr["T1_P50"],        mask_T1)
m_T2r = metrics(enr["t2_actual_min"],  enr["T2_P50"],        mask_T2)
m_T2c = metrics(enr["t2_actual_min"],  enr["T2_P50_CORR"],   mask_T2)
m_T3  = metrics(enr["t3_actual_min"],  enr["T3_P50"],        mask_T3)
m_E0  = metrics(enr["ETA_actual_min"], enr["ETA_P50"],       mask_ETA)
m_EC  = metrics(enr["ETA_actual_min"], enr["ETA_P50_CORR"],  mask_ETA)

diag_raw = pd.DataFrame({"rows":[m_T1[2], m_T2r[2], m_T3[2], m_E0[2]],
                         "MAE":[ m_T1[0], m_T2r[0], m_T3[0], m_E0[0]],
                         "Median_AE":[m_T1[1], m_T2r[1], m_T3[1], m_E0[1]]},
                        index=["T1","T2_raw","T3","ETA_raw"])
diag_cor = pd.DataFrame({"rows":[m_T2c[2], m_EC[2]],
                         "MAE":[ m_T2c[0], m_EC[0]],
                         "Median_AE":[m_T2c[1], m_EC[1]]},
                        index=["T2_corrected","ETA_corrected"])

diag_raw.to_csv(os.path.join(OUTDIR,"eta_leg_diagnostics.csv"))
diag_cor.to_csv(os.path.join(OUTDIR,"eta_leg_diagnostics_corrected.csv"))

for name, col in [("zone_hour_diagnostics.csv","ETA_P50"), ("zone_hour_diagnostics_corrected.csv","ETA_P50_CORR")]:
    if {"zone_name_en","pickup_hr_ksa"}.issubset(enr.columns):
        zxh = (enr.assign(err=lambda d: (d["ETA_actual_min"] - d[col]).abs())
                  .groupby(["zone_name_en","pickup_hr_ksa"])
                  .agg(rows=("err","size"),
                       MAE=("err","mean"),
                       Median_AE=("err","median"))
                  .reset_index()
                  .sort_values(["MAE"], ascending=False))
        zxh.to_csv(os.path.join(OUTDIR, name), index=False)

cov_lines = []
N_MIN = 25
if "t2a_n_cell" in enr.columns:
    frac_good_t2a = float((enr["t2a_n_cell"] >= N_MIN).mean()*100.0)
    cov_lines.append(f"T2a coverage: {frac_good_t2a:.1f}% of orders hit cells with n_cell >= {N_MIN}")
else:
    cov_lines.append("T2a coverage: n/a")
if "t2b_n_cell" in enr.columns:
    frac_cell_t2b = float((enr["t2b_n_cell"] >= N_MIN).mean()*100.0)
    cov_lines.append(f"T2b (shop×hour×dow) direct-cell coverage: {frac_cell_t2b:.1f}% with n_cell >= {N_MIN}")
else:
    cov_lines.append("T2b coverage: n/a")
bias_active = (enr["T2_bias_corr"].abs() >= BIAS_MIN_ABS).mean()*100.0
cov_lines.append(f"T2 bias activation rate: {bias_active:.1f}% of orders (|bias| >= {BIAS_MIN_ABS} & N>= {BIAS_MIN_N})")
with open(os.path.join(OUTDIR,"coverage_report.txt"), "w") as f:
    f.write("\n".join(cov_lines) + "\n")

keep = ["order_id","zone_name_en","pickup_hr_ksa","is_peak","dow","stack_bkt",
        "T1_P50","t1_actual_min",
        "T2a_P50","t2a_ratio_p50","t2a_ratio_src","t2a_n_cell",
        "T2b_base_P50","T2b_resid_P50","T2b_resid_gated","T2b_P50","t2b_n_cell",
        "t2a_actual_min","t2b_actual_min",
        "T2_P50","T2_bias_corr","T2_P50_CORR","t2_actual_min",
        "t3_coord_flag","t3_coord_reason","t3_dist_haversine_km","t3_pred_mode",
        "t3_ratio_source","t3_ratio_source_fallback",
        "T3_osrmratio_P50","T3_base_P50","T4_dwell_P50","T3_P50","t3_actual_min",
        "ETA_P50","ETA_P50_CORR","ETA_actual_min"]
keep = [c for c in keep if c in enr.columns]
enr[keep].to_csv(os.path.join(OUTDIR,"order_level_predictions.csv"), index=False)

print("\n=== ETA v5.7.5 (P50) — RAW ===")
print(diag_raw.to_string())
print("\n=== ETA v5.7.5 (P50) — CORRECTED ===")
print(diag_cor.to_string())
print(f"\nArtifacts → {os.path.abspath(OUTDIR)}")
