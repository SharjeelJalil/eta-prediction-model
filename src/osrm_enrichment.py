#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSRM Enrichment Pipeline for ETA Modeling
==========================================
Fetches driving duration and distance from a local OSRM (Open Source Routing
Machine) server for every order's courier→vendor and vendor→customer route.

Computes calibration ratio tables (actual_time / osrm_time) segmented by
zone × hour × distance bin, which downstream models use to translate ideal
OSRM estimates into realistic delivery predictions.

Pipeline stages:
  1. Parse and validate timestamps for T1, T2, T3 segments
  2. Validate and fix swapped lat/lon coordinates
  3. Deduplicate route pairs and batch-query OSRM with multithreading
  4. Compute actual-over-OSRM ratios per order
  5. Build calibration tables aggregated by zone × hour (T1) and
     zone × hour × distance bin (T2, T3)

Outputs:
  - eta_osrm_enriched.csv          — per-order enriched file
  - eta_osrm_calibration_t1.csv    — T1 calibration (zone × hour)
  - eta_osrm_calibration_t2.csv    — T2 calibration (zone × hour × dist bin)
  - eta_osrm_calibration_t3.csv    — T3 calibration (zone × hour × dist bin)

Requires: Local OSRM server running on the configured port.
"""

import pandas as pd, numpy as np, requests, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pytz

# ── CONFIG ───────────────────────────────────────────────────────────────
OSRM_BASE   = "http://localhost:5001"     # ← your OSRM
INPUT_CSV   = "ETAOrdersJulSept.csv"
OUT_ENRICH  = "eta_osrm_jul_enriched.csv"
OUT_CALIB_T1 = "eta_osrm__jul_calibration_t1.csv"
OUT_CALIB_T2 = "eta_osrm_jul_calibration_t2.csv"
OUT_CALIB_T3 = "eta_osrm_jul_calibration_t3.csv"

ROUND_PLACES = 5
MAX_WORKERS  = 16
TIMEOUT_S    = 4
RETRIES      = 2

# Travel-time sanity (minutes)
MIN_T1 = 0.1
MAX_T1 = 60
MIN_T2 = 0.2
MAX_T2 = 60
MIN_T3 = 0.5
MAX_T3 = 150

TZ_KSA = pytz.timezone("Asia/Riyadh")

# ── HELPERS ──────────────────────────────────────────────────────────────
def parse_ksa(s):
    if pd.isna(s): return pd.NaT
    dt = pd.to_datetime(s, errors="coerce")
    if dt is pd.NaT: return pd.NaT
    if dt.tzinfo is None:
        try: dt = TZ_KSA.localize(dt)
        except Exception: return pd.NaT
    return dt

def coalesce_ts(df, cols):
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in cols:
        if c in df.columns:
            out = out.where(out.notna(), df[c])
    return out

def osrm_route(session, lon1, lat1, lon2, lat2, retries=RETRIES):
    url = (f"{OSRM_BASE}/route/v1/driving/"
           f"{lon1:.5f},{lat1:.5f};{lon2:.5f},{lat2:.5f}"
           f"?overview=false&alternatives=false&steps=false&annotations=false&radiuses=5000;5000")
    for attempt in range(retries+1):
        try:
            r = session.get(url, timeout=TIMEOUT_S)
            j = r.json()
            if j.get("code") == "Ok" and j.get("routes"):
                d = float(j["routes"][0]["duration"])  # sec
                m = float(j["routes"][0]["distance"])  # m
                return d, m
        except Exception:
            pass
        time.sleep(0.3*(attempt+1))
    return np.nan, np.nan

def make_key(lat1, lon1, lat2, lon2):
    return f"{lat1:.{ROUND_PLACES}f},{lon1:.{ROUND_PLACES}f}→{lat2:.{ROUND_PLACES}f},{lon2:.{ROUND_PLACES}f}"

# ── LOAD DATA ────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")
print(f"ROWS (raw): {len(df):,}")

# Parse timestamps
for c in [
    "order_created_ksa","courier_assignment_accepted_timestamp",
    "pickup_departure_ksa","pickup_btn_clicked_ksa","pickup_arrived_ksa",
    "dropoff_arrived_ksa","order_delivered_ksa","dropoff_btn_clicked_ksa"
]:
    if c in df.columns:
        df[c] = df[c].apply(parse_ksa)

# ── T1 ACTUAL (Created → Accepted) ──────────────────────────────────────
df["t1_start_ts"] = df["order_created_ksa"]
df["t1_end_ts"]   = df["courier_assignment_accepted_timestamp"]

mask_t1 = df["t1_start_ts"].notna() & df["t1_end_ts"].notna()
df["t1_actual_min"] = (df["t1_end_ts"] - df["t1_start_ts"]).dt.total_seconds()/60.0
df.loc[~mask_t1, "t1_actual_min"] = np.nan
df = df[(df["t1_actual_min"].isna()) | df["t1_actual_min"].between(MIN_T1, MAX_T1)].copy()

# ── T2 ACTUAL (Accepted → Pickup Arrived) ───────────────────────────────
df["t2_start_ts"] = df["courier_assignment_accepted_timestamp"]
df["t2_end_ts"]   = df["pickup_arrived_ksa"]

mask_t2 = df["t2_start_ts"].notna() & df["t2_end_ts"].notna()
df["t2_actual_min"] = (df["t2_end_ts"] - df["t2_start_ts"]).dt.total_seconds()/60.0
df.loc[~mask_t2, "t2_actual_min"] = np.nan
df = df[(df["t2_actual_min"].isna()) | df["t2_actual_min"].between(MIN_T2, MAX_T2)].copy()

# ── T3 ACTUAL (Pickup → Dropoff) ────────────────────────────────────────
df["t3_start_ts"] = coalesce_ts(df, ["pickup_departure_ksa","pickup_btn_clicked_ksa","pickup_arrived_ksa"])
df["t3_end_ts"]   = coalesce_ts(df, ["dropoff_arrived_ksa","order_delivered_ksa","dropoff_btn_clicked_ksa"])

mask_t3 = df["t3_start_ts"].notna() & df["t3_end_ts"].notna()
df["t3_actual_min"] = (df["t3_end_ts"] - df["t3_start_ts"]).dt.total_seconds()/60.0
df.loc[~mask_t3, "t3_actual_min"] = np.nan
df = df[(df["t3_actual_min"].isna()) | df["t3_actual_min"].between(MIN_T3, MAX_T3)].copy()

print(f"… rows with valid T1 times: {df['t1_actual_min'].notna().sum():,}")
print(f"… rows with valid T2 times: {df['t2_actual_min'].notna().sum():,}")
print(f"… rows with valid T3 times: {df['t3_actual_min'].notna().sum():,}")

# ── COORD SANITY ────────────────────────────────────────────────────────
coord_cols = [
    "pickup_lat","pickup_lon","dropoff_lat","dropoff_lon",
    "courier_latitude_at_assignment","courier_longitude_at_assignment"
]
for c in coord_cols:
    if c not in df.columns:
        raise SystemExit(f"Missing coord column: {c}")

# Fix swapped lat/lon
swap_cols = [
    ("pickup_lat","pickup_lon"),
    ("dropoff_lat","dropoff_lon"),
    ("courier_latitude_at_assignment","courier_longitude_at_assignment")
]
for lat_c, lon_c in swap_cols:
    mask = (df[lat_c].abs() > 40) & (df[lon_c].abs() < 40)
    df.loc[mask, [lat_c,lon_c]] = df.loc[mask, [lon_c,lat_c]].values

# Round
for c in coord_cols:
    df[c] = df[c].astype(float).round(ROUND_PLACES)

# ── ROUTE KEYS (T2 + T3) ────────────────────────────────────────────────
df["t2_pair_key"] = [make_key(a,b,c,d) for a,b,c,d in 
                     df[["courier_latitude_at_assignment","courier_longitude_at_assignment","pickup_lat","pickup_lon"]].to_numpy()]
df["t3_pair_key"] = [make_key(a,b,c,d) for a,b,c,d in 
                     df[["pickup_lat","pickup_lon","dropoff_lat","dropoff_lon"]].to_numpy()]

pairs_t2 = df[["t2_pair_key","courier_latitude_at_assignment","courier_longitude_at_assignment","pickup_lat","pickup_lon"]] \
             .drop_duplicates().reset_index(drop=True)
pairs_t3 = df[["t3_pair_key","pickup_lat","pickup_lon","dropoff_lat","dropoff_lon"]] \
             .drop_duplicates().reset_index(drop=True)

print(f"… unique T2 pairs: {len(pairs_t2):,}")
print(f"… unique T3 pairs: {len(pairs_t3):,}")

# ── OSRM CALLS (T2 + T3) ────────────────────────────────────────────────
session = requests.Session()

def worker_t2(row):
    key, clat, clon, plat, plon = row
    sec, meters = osrm_route(session, clon, clat, plon, plat)
    return key, sec, meters

def worker_t3(row):
    key, plat, plon, dlat, dlon = row
    sec, meters = osrm_route(session, plon, plat, dlon, dlat)
    return key, sec, meters

def run_osrm(pairs, worker, desc):
    results = {}
    rows = [tuple(r) for r in pairs.itertuples(index=False, name=None)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="route"):
            key, sec, meters = fut.result()
            results[key] = (sec, meters)
    return results

results_t2 = run_osrm(pairs_t2, worker_t2, "OSRM T2")
results_t3 = run_osrm(pairs_t3, worker_t3, "OSRM T3")

# ── ATTACH RESULTS ──────────────────────────────────────────────────────
df["t2_osrm_sec"] = df["t2_pair_key"].map(lambda k: results_t2.get(k, (np.nan,np.nan))[0])
df["t2_osrm_m"]   = df["t2_pair_key"].map(lambda k: results_t2.get(k, (np.nan,np.nan))[1])
df["t3_osrm_sec"] = df["t3_pair_key"].map(lambda k: results_t3.get(k, (np.nan,np.nan))[0])
df["t3_osrm_m"]   = df["t3_pair_key"].map(lambda k: results_t3.get(k, (np.nan,np.nan))[1])

df["t2_osrm_duration_min"] = df["t2_osrm_sec"] / 60.0
df["t2_osrm_distance_km"]  = df["t2_osrm_m"] / 1000.0
df["t3_osrm_duration_min"] = df["t3_osrm_sec"] / 60.0
df["t3_osrm_distance_km"]  = df["t3_osrm_m"] / 1000.0

df["ratio_actual_over_osrm_t2"] = df["t2_actual_min"] / df["t2_osrm_duration_min"]
df["ratio_actual_over_osrm_t3"] = df["t3_actual_min"] / df["t3_osrm_duration_min"]

# ── SAVE ENRICHED ──────────────────────────────────────────────────────
df.to_csv(OUT_ENRICH, index=False)
print(f"✓ Saved enriched orders → {OUT_ENRICH}  (rows: {len(df):,})")

# ── CALIBRATION FUNCTIONS ───────────────────────────────────────────────
def build_calibration_t1(df, out_file):
    df["pickup_hr_ksa"] = df["t1_start_ts"].dt.tz_convert("Asia/Riyadh").dt.hour
    calib = (df.dropna(subset=["pickup_hr_ksa","zone_name_en","t1_actual_min"])
               .groupby(["pickup_hr_ksa","zone_name_en"])
               .agg(
                   t1_p50=("t1_actual_min", lambda s: float(np.nanmedian(s))),
                   t1_p85=("t1_actual_min", lambda s: float(np.nanquantile(s, 0.85))),
                   n=("t1_actual_min","size")
               )
               .reset_index()
               .sort_values(["pickup_hr_ksa","zone_name_en"]))
    calib.to_csv(out_file, index=False)
    print(f"✓ Saved T1 calibration → {out_file} (rows: {len(calib):,})")

def build_calibration(df, actual_col, osrm_col, ratio_col, out_file, label):
    edges  = [0,1,2,4,6,8,12,20,1e9]
    labels = ["<1","1-2","2-4","4-6","6-8","8-12","12-20","20+"]
    df["dist_bin"] = pd.cut(df[osrm_col], bins=edges, labels=labels, include_lowest=True, right=False)
    df["pickup_hr_ksa"] = df["t3_start_ts"].dt.tz_convert("Asia/Riyadh").dt.hour
    calib = (df.dropna(subset=["pickup_hr_ksa","zone_name_en","dist_bin",ratio_col])
               .groupby(["pickup_hr_ksa","zone_name_en","dist_bin"])
               .agg(
                   ratio_p50=(ratio_col, lambda s: float(np.nanmedian(s))),
                   ratio_p85=(ratio_col, lambda s: float(np.nanquantile(s, 0.85))),
                   n=(ratio_col, "size"),
                   osrm_km_p50=(osrm_col, "median"),
                   actual_min_p50=(actual_col, "median")
               )
               .reset_index()
               .sort_values(["pickup_hr_ksa","zone_name_en","dist_bin"]))
    calib.to_csv(out_file, index=False)
    print(f"✓ Saved {label} calibration → {out_file} (rows: {len(calib):,})")

# ── CALIBRATION (T1 + T2 + T3) ─────────────────────────────────────────
build_calibration_t1(df, OUT_CALIB_T1)
build_calibration(df, "t2_actual_min", "t2_osrm_distance_km", "ratio_actual_over_osrm_t2", OUT_CALIB_T2, "T2")
build_calibration(df, "t3_actual_min", "t3_osrm_distance_km", "ratio_actual_over_osrm_t3", OUT_CALIB_T3, "T3")
