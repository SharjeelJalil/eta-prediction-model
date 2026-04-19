#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETA Model v5.1 — Smoothed Ratios + Prep-Aware T2 Residual + T4 Dwell
=====================================================================
End-to-end ETA prediction combining:
  - T1: Gradient-boosted quantile regression for courier acceptance time
  - T2: OSRM × smoothed ratio baseline + ML residual correction with
         prep-time features + peak-hour guardrails for worst-performing zones
  - T3: OSRM × smoothed ratio for transit time
  - T4: Zone × hour smoothed dwell time (arrival → delivered)

Key design choices in this version:
  - Hierarchical Bayesian smoothing on OSRM ratios with configurable
    shrinkage strength (N0 parameter)
  - Prep-time features (shop-level median, IQR, count) integrated into
    the T2 residual model to capture vendor-specific cooking patterns
  - Data-driven peak guardrails: blends zone × peak × stacking bucket
    mean residuals for the worst-performing zones during rush hours


import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# ---------------- CONFIG ----------------
ENR_CSV  = "eta_osrm_enriched.csv"
T2_CALIB = "eta_osrm_calibration_t2.csv"
T3_CALIB = "eta_osrm_calibration_t3.csv"
ORD_PT   = "OrderLevelPT.csv"
SHOP_PT  = "ShopLevelPT.csv"
VERT_PT  = "VerticalLevelPT.csv"

OUTDIR   = "./eta_outputs_v5_1"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
N0_T2 = 50   # smoothing strength for T2 ratios
N0_T3 = 30   # smoothing strength for T3 ratios
N0_DW = 80   # smoothing strength for dwell medians (more aggressive)

MIN_PRED_T2 = 0.5
MAX_PRED_T2 = 240.0
MIN_PRED_T3 = 0.5
MAX_PRED_T3 = 180.0
MIN_PRED_T4 = 0.0
MAX_PRED_T4 = 60.0

PEAK_HOURS = set(range(14, 23))  # 14..22 inclusive

DIST_EDGES  = [0,1,2,4,6,8,12,20,1e9]
DIST_LABELS = ["<1","1-2","2-4","4-6","6-8","8-12","12-20","20+"]

WORST_ZONES = {
    "South West Riyadh","West Riyadh","West central Riyadh",
    "North West Riyadh","Central Riyadh","North Riyadh"
}

# ---------------- UTILS ----------------
def dist_label_for(km: float) -> Optional[str]:
    if pd.isna(km): return None
    for lo, hi, lab in zip(DIST_EDGES[:-1], DIST_EDGES[1:], DIST_LABELS):
        if lo <= km < hi: return lab
    return "20+"

def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)

def median_with_weights(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    d = df[[value_col, weight_col]].dropna()
    if d.empty:
        return float("nan")
    d = d.sort_values(value_col)
    csum = d[weight_col].cumsum()
    cutoff = d[weight_col].sum() / 2.0
    idx = (csum >= cutoff).idxmax()
    return float(d.loc[idx, value_col])

def build_smoothed_ratios(raw: pd.DataFrame, n0: int, ratio_col="ratio_p50") -> pd.DataFrame:
    need = ["zone_name_en","pickup_hr_ksa","dist_bin",ratio_col,"n"]
    miss = [c for c in need if c not in raw.columns]
    if miss: raise ValueError(f"Calibration missing columns: {miss}")

    df = raw.copy()
    df["pickup_hr_ksa"] = pd.to_numeric(df["pickup_hr_ksa"], errors="coerce").astype("Int64")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["is_peak"] = df["pickup_hr_ksa"].apply(lambda x: int(x in PEAK_HOURS) if pd.notna(x) else 0)

    # Parents
    hd  = (df.groupby(["pickup_hr_ksa","dist_bin"], dropna=False)
             .apply(lambda g: pd.Series({"ratio_hd": median_with_weights(g, ratio_col, "n"),
                                         "n_hd": int(g["n"].sum())}))
             .reset_index())
    ipd = (df.groupby(["is_peak","dist_bin"], dropna=False)
             .apply(lambda g: pd.Series({"ratio_ipd": median_with_weights(g, ratio_col, "n"),
                                         "n_ipd": int(g["n"].sum())}))
             .reset_index())
    h   = (df.groupby(["pickup_hr_ksa"], dropna=False)
             .apply(lambda g: pd.Series({"ratio_h": median_with_weights(g, ratio_col, "n"),
                                         "n_h": int(g["n"].sum())}))
             .reset_index())
    ip  = (df.groupby(["is_peak"], dropna=False)
             .apply(lambda g: pd.Series({"ratio_ip": median_with_weights(g, ratio_col, "n"),
                                         "n_ip": int(g["n"].sum())}))
             .reset_index())
    global_ratio = median_with_weights(df, ratio_col, "n")
    global_row = {"ratio_g": float(global_ratio), "n_g": int(df["n"].sum())}

    m = (df.merge(hd, on=["pickup_hr_ksa","dist_bin"], how="left")
           .merge(ipd, on=["is_peak","dist_bin"], how="left")
           .merge(h, on=["pickup_hr_ksa"], how="left")
           .merge(ip, on=["is_peak"], how="left"))

    def choose_parent(row):
        for rc, nc in [("ratio_hd","n_hd"), ("ratio_ipd","n_ipd"), ("ratio_h","n_h"), ("ratio_ip","n_ip")]:
            r = row.get(rc, np.nan); n = row.get(nc, 0)
            if pd.notna(r) and (n > 0):
                return float(r), int(n), rc
        return float(global_row["ratio_g"]), int(global_row["n_g"]), "ratio_g"

    parents = m.apply(choose_parent, axis=1, result_type="expand")
    m["parent_ratio"] = parents[0].astype(float)
    m["parent_n"]     = parents[1].astype(int)
    m["parent_level"] = parents[2].astype(str)

    m["ratio_cell"] = pd.to_numeric(m[ratio_col], errors="coerce")
    m["n_cell"]     = m["n"].astype(int)
    m["n0"]         = n0
    m["ratio_smoothed"] = (m["n_cell"] * m["ratio_cell"] + n0 * m["parent_ratio"]) / (m["n_cell"] + n0)

    keep = ["zone_name_en","pickup_hr_ksa","dist_bin","is_peak",
            "ratio_cell","n_cell","parent_ratio","parent_n","parent_level","n0","ratio_smoothed"]
    return m[keep].copy()

def lookup_ratio(zone: str, hour: int, dist_bin: str, table: pd.DataFrame) -> Tuple[float, str]:
    hit = table[(table["zone_name_en"]==zone) &
                (table["pickup_hr_ksa"]==hour) &
                (table["dist_bin"]==dist_bin)]
    if len(hit):
        r = float(hit["ratio_smoothed"].iloc[0])
        src = "zone_hour_dist_smoothed" if hit["n_cell"].iloc[0] > 0 else f"parent:{hit['parent_level'].iloc[0]}"
        return r, src
    is_peak = int(hour in PEAK_HOURS)
    p = table[(table["is_peak"]==is_peak) & (table["dist_bin"]==dist_bin)]
    if len(p):
        return float(p["ratio_smoothed"].median()), "fallback:is_peak_dist"
    h = table[table["pickup_hr_ksa"]==hour]
    if len(h):
        return float(h["ratio_smoothed"].median()), "fallback:hour"
    return float(table["ratio_smoothed"].median()), "fallback:global"

def train_t1(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    cat = [c for c in ["sub_vertical_name","zone_name_en","business_branch_name","shop_id","mbo_chain_id"] if c in df.columns]
    num = [c for c in ["stacking_order_count","accepted_courier_count","rejected_courier_count",
                       "pickup_lat","pickup_lon","dropoff_lat","dropoff_lon",
                       "order_hr_ksa","order_dow_ksa"] if c in df.columns]
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

def stacking_bucket(x):
    try: return int(float(x) >= 1.0)
    except Exception: return 0

# ---------------- LOAD ----------------
enr   = pd.read_csv(ENR_CSV, low_memory=False)
t2raw = pd.read_csv(T2_CALIB, low_memory=False)
t3raw = pd.read_csv(T3_CALIB, low_memory=False)
ordpt = pd.read_csv(ORD_PT, low_memory=False)
shoppt= pd.read_csv(SHOP_PT, low_memory=False)
vertpt= pd.read_csv(VERT_PT, low_memory=False)

# --------- HARDEN TIMESTAMP PARSING ---------
enr.columns = enr.columns.str.strip()
ts_cols = [c for c in enr.columns if c.endswith("_ksa")]
for c in ts_cols:
    s = pd.to_datetime(enr[c], errors="coerce", utc=False)
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        pass
    enr[c] = s

# Derive features safely
if "order_created_ksa" in enr.columns:
    enr["order_hr_ksa"]  = enr["order_created_ksa"].dt.hour
    enr["order_dow_ksa"] = enr["order_created_ksa"].dt.weekday
else:
    enr["order_hr_ksa"], enr["order_dow_ksa"] = np.nan, np.nan

if ("pickup_hr_ksa" not in enr.columns) or (enr["pickup_hr_ksa"].isna().all()):
    if "pickup_departure_ksa" in enr.columns:
        enr["pickup_hr_ksa"] = enr["pickup_departure_ksa"].dt.hour
    else:
        enr["pickup_hr_ksa"] = np.nan

enr["is_peak"] = enr["pickup_hr_ksa"].apply(lambda h: int(h in PEAK_HOURS) if pd.notna(h) else 0)
enr["dow"]      = enr["order_dow_ksa"]
# --------------------------------------------

# Ensure OSRM numeric fields
for c in ["t2_osrm_duration_min","t2_osrm_distance_km","t3_osrm_duration_min","t3_osrm_distance_km"]:
    if c in enr.columns:
        enr[c] = pd.to_numeric(enr[c], errors="coerce")

# ---------------- RECOMPUTE LEG & ETA ACTUALS CONSISTENTLY ----------------
# Make T3/ETA use the SAME end point: dropoff_arrived_ksa if present, else order_delivered_ksa
end_for_t3 = enr["dropoff_arrived_ksa"].fillna(enr["order_delivered_ksa"])
enr["t1_actual_min"] = (enr["last_accepted_ksa"]   - enr["order_created_ksa"]).dt.total_seconds()/60.0
enr["t2_actual_min"] = (enr["pickup_departure_ksa"]- enr["last_accepted_ksa"]).dt.total_seconds()/60.0
enr["t3_actual_min"] = (end_for_t3                 - enr["pickup_departure_ksa"]).dt.total_seconds()/60.0
enr["ETA_actual_min"]= (end_for_t3                 - enr["order_created_ksa"]).dt.total_seconds()/60.0

# ---------------- BUILD SMOOTHED RATIO TABLES (T2/T3) ----------------
t2_smooth = build_smoothed_ratios(t2raw, n0=N0_T2, ratio_col="ratio_p50")
t3_smooth = build_smoothed_ratios(t3raw, n0=N0_T3, ratio_col="ratio_p50")
t2_smooth.to_csv(os.path.join(OUTDIR,"t2_smoothed_ratios.csv"), index=False)
t3_smooth.to_csv(os.path.join(OUTDIR,"t3_smoothed_ratios.csv"), index=False)

# ---------------- DWELL (T4) MODEL: arrival → delivered ----------------
# Compute dwell only for orders that HAVE both timestamps
have_arrived   = enr["dropoff_arrived_ksa"].notna()
have_delivered = enr["order_delivered_ksa"].notna()
mask_dw_train  = have_arrived & have_delivered
enr["dwell_min"] = (enr["order_delivered_ksa"] - enr["dropoff_arrived_ksa"]).dt.total_seconds()/60.0
# Guard against negative/huge
enr.loc[mask_dw_train, "dwell_min"] = enr.loc[mask_dw_train, "dwell_min"].clip(0, 60)

# Build zone×hour smoothed dwell medians
dw = (enr.loc[mask_dw_train, ["zone_name_en","pickup_hr_ksa","dwell_min"]]
         .assign(n=1))
# hour medians
dw_hour = (dw.groupby("pickup_hr_ksa")
             .agg(dwell_h=("dwell_min","median"),
                  n_h=("n","sum"))
             .reset_index())
# global
dwell_g = float(dw["dwell_min"].median()) if mask_dw_train.any() else 0.0

# attach hour parent to each row for smoothing
dw = dw.merge(dw_hour, on="pickup_hr_ksa", how="left")
# compute smoothed per (zone,hour)
zxh = (dw.groupby(["zone_name_en","pickup_hr_ksa"])
         .agg(dwell_cell=("dwell_min","median"),
              n_cell=("n","sum"),
              dwell_parent=("dwell_h","median"),
              n_parent=("n_h","median"))
         .reset_index())
zxh["dwell_parent"] = zxh["dwell_parent"].fillna(dwell_g)
zxh["n_parent"] = zxh["n_parent"].fillna(dw["n"].sum())
zxh["n0"] = N0_DW
zxh["dwell_smoothed"] = (zxh["n_cell"] * zxh["dwell_cell"] + N0_DW * zxh["dwell_parent"]) / (zxh["n_cell"] + N0_DW)

# fallback table for hour/global
dwell_hour_table = dw_hour.copy()
dwell_hour_table["dwell_h"] = dwell_hour_table["dwell_h"].fillna(dwell_g)

zxh.to_csv(os.path.join(OUTDIR,"dwell_smoothed.csv"), index=False)

def dwell_lookup(zone, hour):
    hit = zxh[(zxh["zone_name_en"]==zone) & (zxh["pickup_hr_ksa"]==hour)]
    if len(hit):
        return float(hit["dwell_smoothed"].iloc[0]), "zone_hour_dwell"
    hh = dwell_hour_table[dwell_hour_table["pickup_hr_ksa"]==hour]
    if len(hh):
        return float(hh["dwell_h"].iloc[0]), "hour_dwell"
    return float(dwell_g), "global_dwell"

# ---------------- T1 (ML P50) ----------------
t1_df = enr.dropna(subset=["t1_actual_min"]).query("0 <= t1_actual_min <= 60").copy()
def train_t1(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    cat = [c for c in ["sub_vertical_name","zone_name_en","business_branch_name","shop_id","mbo_chain_id"] if c in df.columns]
    num = [c for c in ["stacking_order_count","accepted_courier_count","rejected_courier_count",
                       "pickup_lat","pickup_lon","dropoff_lat","dropoff_lon",
                       "order_hr_ksa","order_dow_ksa"] if c in df.columns]
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

t1_model, t1_cat, t1_num = train_t1(t1_df)
joblib.dump(t1_model, os.path.join(OUTDIR,"t1_p50.joblib"))
X1 = enr.reindex(columns=t1_cat+t1_num, fill_value=np.nan)
enr["T1_P50"] = t1_model.predict(X1)

# ---------------- T2 BASELINE (OSRM × Smoothed Ratio) ----------------
enr["t2_dist_bin"] = enr["t2_osrm_distance_km"].map(dist_label_for)
def _ratio_for_t2(z, h, b):
    r, src = lookup_ratio(z, int(h) if pd.notna(h) else -1, b, t2_smooth)
    return r, src
t2_vals = [_ratio_for_t2(z, h, b) for z,h,b in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["t2_dist_bin"])]
enr["t2_ratio_p50"]    = [x[0] for x in t2_vals]
enr["t2_ratio_source"] = [x[1] for x in t2_vals]
enr["T2_osrm_ratio_pred"] = enr["t2_osrm_duration_min"] * enr["t2_ratio_p50"]

# ---------------- PREP FEATURES (for T2 residual) ----------------
enr = enr.merge(shoppt.rename(columns={"shop_id":"mbo_chain_id","prep_time_min":"prep_shop_mean"}),
                on="mbo_chain_id", how="left")
enr = enr.merge(vertpt.rename(columns={"sub_vertical_detail":"sub_vertical_name","prep_time_min":"prep_vert_mean"}),
                on="sub_vertical_name", how="left")

if "order_id" in ordpt.columns and "prep_time_min" in ordpt.columns:
    ordpt_small = ordpt[["order_id","prep_time_min"]].dropna()
    tmp = enr[["order_id","mbo_chain_id"]].dropna().astype({"mbo_chain_id": str})
    ord_with_shop = ordpt_small.merge(tmp, on="order_id", how="inner")
    shop_stats = (ord_with_shop.groupby("mbo_chain_id")["prep_time_min"]
                  .agg(prep_shop_p50="median",
                       prep_shop_p25=lambda s: float(np.percentile(s,25)),
                       prep_shop_p75=lambda s: float(np.percentile(s,75)),
                       prep_shop_cnt="count").reset_index())
    shop_stats["prep_shop_iqr"] = shop_stats["prep_shop_p75"] - shop_stats["prep_shop_p25"]
    enr = enr.merge(shop_stats[["mbo_chain_id","prep_shop_p50","prep_shop_iqr","prep_shop_cnt"]],
                    on="mbo_chain_id", how="left")

# ---------------- T2 RESIDUAL MODEL (P50) ----------------
enr["t2_actual_min"] = pd.to_numeric(enr.get("t2_actual_min", np.nan), errors="coerce")
mask_t2 = enr["t2_actual_min"].between(0.5, 240) & enr["T2_osrm_ratio_pred"].notna()
t2_train = enr[mask_t2].copy()
t2_train["t2_resid"] = t2_train["t2_actual_min"] - t2_train["T2_osrm_ratio_pred"]
t2_train["t2_resid"] = winsorize(t2_train["t2_resid"], -15.0, 45.0)

t2_cat = [c for c in [
    "zone_name_en","sub_vertical_name","business_branch_name","shop_id","mbo_chain_id"
] if c in t2_train.columns]
t2_num = [c for c in [
    "pickup_hr_ksa","is_peak","dow",
    "stacking_order_count","accepted_courier_count","rejected_courier_count",
    "t2_osrm_distance_km","t2_osrm_duration_min",
    "prep_shop_mean","prep_vert_mean","prep_shop_p50","prep_shop_iqr","prep_shop_cnt"
] if c in t2_train.columns]

t2_pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), t2_cat),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), t2_num),
])

t2_resid_model = Pipeline([
    ("prep", t2_pre),
    ("gb", GradientBoostingRegressor(
        loss="quantile", alpha=0.5,
        max_depth=4, n_estimators=600, min_samples_leaf=25, random_state=SEED
    ))
])
t2_resid_model.fit(t2_train[t2_cat+t2_num], t2_train["t2_resid"])
joblib.dump(t2_resid_model, os.path.join(OUTDIR,"t2_resid_p50.joblib"))

X2 = enr.reindex(columns=t2_cat+t2_num, fill_value=np.nan)
enr["T2_resid_P50"] = t2_resid_model.predict(X2)

# Peak guardrail (data-driven)
t2_train["stacking_bkt"] = t2_train["stacking_order_count"].apply(stacking_bucket)
guard = (t2_train.groupby(["zone_name_en","is_peak","stacking_bkt"])["t2_resid"]
         .mean().reset_index().rename(columns={"t2_resid":"guard_mean"}))
enr["stacking_bkt"] = enr["stacking_order_count"].apply(stacking_bucket)
enr = enr.merge(guard, on=["zone_name_en","is_peak","stacking_bkt"], how="left")
lambda_guard = 0.7
enr["guard_adj"] = np.where(
    (enr["zone_name_en"].isin(WORST_ZONES)) & (enr["is_peak"]==1) & (enr["guard_mean"]>0),
    lambda_guard * enr["guard_mean"], 0.0
)

# Final T2
enr["T2_P50"] = (enr["T2_osrm_ratio_pred"] + enr["T2_resid_P50"] + enr["guard_adj"]).clip(MIN_PRED_T2, MAX_PRED_T2)

# ---------------- T3 (OSRM × Smoothed Ratio) ----------------
enr["t3_dist_bin"] = enr["t3_osrm_distance_km"].map(dist_label_for)
def _ratio_for_t3(z, h, b):
    r, src = lookup_ratio(z, int(h) if pd.notna(h) else -1, b, t3_smooth)
    return r, src
t3_vals = [_ratio_for_t3(z, h, b) for z,h,b in zip(enr["zone_name_en"], enr["pickup_hr_ksa"], enr["t3_dist_bin"])]
enr["t3_ratio_p50"]    = [x[0] for x in t3_vals]
enr["t3_ratio_source"] = [x[1] for x in t3_vals]
enr["T3_base_P50"]     = (enr["t3_osrm_duration_min"] * enr["t3_ratio_p50"]).clip(MIN_PRED_T3, MAX_PRED_T3)

# ---------------- T4 (arrival→delivered dwell) for orders w/o arrival ----------------
needs_dwell = enr["dropoff_arrived_ksa"].isna() & enr["order_delivered_ksa"].notna()
dwell_pred = [0.0]*len(enr)
dwell_src  = ["none"]*len(enr)
for i, (need, z, h) in enumerate(zip(needs_dwell, enr["zone_name_en"], enr["pickup_hr_ksa"])):
    if not need:
        dwell_pred[i] = 0.0
        dwell_src[i]  = "none"
    else:
        r, src = dwell_lookup(z, int(h) if pd.notna(h) else -1)
        dwell_pred[i] = float(np.clip(r, MIN_PRED_T4, MAX_PRED_T4))
        dwell_src[i]  = src
enr["T4_dwell_P50"] = dwell_pred
enr["dwell_source"] = dwell_src

# Final T3 = base travel + dwell (only when arrival missing)
enr["T3_P50"] = enr["T3_base_P50"] + enr["T4_dwell_P50"]

# ---------------- ETA & DIAGNOSTICS ----------------
enr["ETA_P50"] = enr["T1_P50"] + enr["T2_P50"] + enr["T3_P50"]

def metrics(actual: pd.Series, pred: pd.Series, mask: pd.Series):
    e = (actual - pred).abs()
    e = e[mask]
    if e.empty: return (np.nan, np.nan, 0)
    return (float(e.mean()), float(e.median()), int(mask.sum()))

m_T1 = metrics(enr["t1_actual_min"], enr["T1_P50"], enr["t1_actual_min"].between(0,60))
m_T2 = metrics(enr["t2_actual_min"], enr["T2_P50"], enr["t2_actual_min"].between(0.5,240))
m_T3 = metrics(enr["t3_actual_min"], enr["T3_P50"], enr["t3_actual_min"].between(0.5,180))
m_ETA= metrics(enr["ETA_actual_min"], enr["ETA_P50"], enr["ETA_actual_min"].between(10,300))

diag = pd.DataFrame({
    "rows":[m_T1[2], m_T2[2], m_T3[2], m_ETA[2]],
    "MAE":[m_T1[0], m_T2[0], m_T3[0], m_ETA[0]],
    "Median_AE":[m_T1[1], m_T2[1], m_T3[1], m_ETA[1]],
}, index=["T1","T2","T3","ETA"])
diag.to_csv(os.path.join(OUTDIR,"eta_leg_diagnostics.csv"))

# Zone×hour diagnostics
zxh = (enr.assign(err=lambda d: (d["ETA_actual_min"] - d["ETA_P50"]).abs())
          .groupby(["zone_name_en","pickup_hr_ksa"])
          .agg(rows=("err","size"),
               MAE=("err","mean"),
               Median_AE=("err","median"))
          .reset_index()
          .sort_values(["MAE"], ascending=False))
zxh.to_csv(os.path.join(OUTDIR,"zone_hour_diagnostics.csv"), index=False)

# Sources per order
pd.DataFrame({
    "order_id": enr["order_id"],
    "t2_ratio_source": enr["t2_ratio_source"],
    "t3_ratio_source": enr["t3_ratio_source"],
    "dwell_source":     enr["dwell_source"]
}).to_csv(os.path.join(OUTDIR,"t2_t3_ratio_sources.csv"), index=False)

# Save order-level predictions
keep = ["order_id","zone_name_en","pickup_hr_ksa","is_peak","dow",
        "T1_P50","t1_actual_min",
        "T2_osrm_ratio_pred","T2_resid_P50","guard_adj","T2_P50","t2_actual_min",
        "T3_base_P50","T4_dwell_P50","T3_P50","t3_actual_min",
        "ETA_P50","ETA_actual_min",
        "t2_ratio_p50","t2_ratio_source","t3_ratio_p50","t3_ratio_source","dwell_source",
        "prep_shop_mean","prep_vert_mean","prep_shop_p50","prep_shop_iqr","prep_shop_cnt"]
enr[keep].to_csv(os.path.join(OUTDIR,"order_level_predictions.csv"), index=False)

print("\n=== ETA v5.1 (P50 only) — Diagnostics ===")
print(diag)
print(f"\nArtifacts → {os.path.abspath(OUTDIR)}")
