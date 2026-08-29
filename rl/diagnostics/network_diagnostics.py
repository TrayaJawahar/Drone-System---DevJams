"""
network_diagnostics.py
----------------------
Prints a detailed diagnostic report of the Geo-Network Map's network quality data.

Usage:
    python -m rl.diagnostics.network_diagnostics

Reports:
  - Column presence and dtype
  - Per-metric: min, max, mean, std, NaN count, zero count, non-zero count
  - What network_score would look like across all cells (from NetworkQualityCalculator)
  - Whether network_mean_quality would be 0 for all cells
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from rl.data.data_loader import load_geo_network_map
from rl.environment.network_quality import NetworkQualityCalculator
from rl.utils.logger import setup_logger

logger = setup_logger("network_diagnostics")

NETWORK_COLS = ["rssi", "rsrp", "sinr", "latency", "packet_loss", "throughput",
                "network_data_confidence", "nearest_tower_distance"]


def _col_stats(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {"present": False}

    s = df[col]
    nan_mask   = s.isna()
    valid      = s.dropna()
    zero_count = int((valid == 0).sum())

    stats = {
        "present":    True,
        "dtype":      str(s.dtype),
        "count":      int(len(s)),
        "nan_count":  int(nan_mask.sum()),
        "non_nan":    int((~nan_mask).sum()),
        "zero_count": zero_count,
        "nonzero":    int(len(valid)) - zero_count,
    }
    if len(valid) > 0:
        stats.update({
            "min":  float(valid.min()),
            "max":  float(valid.max()),
            "mean": float(valid.mean()),
            "std":  float(valid.std()),
            "p25":  float(valid.quantile(0.25)),
            "p50":  float(valid.quantile(0.50)),
            "p75":  float(valid.quantile(0.75)),
        })
    return stats


def run_network_diagnostics(config_path: str = "rl/config/rl_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    map_path  = config["data"]["geo_network_map"]
    meta_path = config["data"]["metadata"]

    if not os.path.exists(map_path):
        print(f"\nERROR: Real Geo-Network Map not found.\nPlease build {map_path} before running diagnostics.\n")
        sys.exit(1)

    print(f"\nLoading map: {map_path}")
    df, metadata = load_geo_network_map(map_path, meta_path)
    print(f"Map loaded: {len(df)} cells, {len(df.columns)} columns\n")

    # ── 1. Column overview ───────────────────────────────────────────────────
    print("=" * 60)
    print("NETWORK QUALITY STATISTICS")
    print("=" * 60)

    for col in NETWORK_COLS:
        stats = _col_stats(df, col)
        if not stats["present"]:
            print(f"\n  {col.upper()}: NOT IN DATASET")
            continue

        print(f"\n  {col.upper()}  (dtype={stats['dtype']})")
        print(f"    NaN count    : {stats['nan_count']} / {stats['count']}  "
              f"({100*stats['nan_count']/max(1,stats['count']):.1f}%)")
        print(f"    Non-NaN count: {stats['non_nan']}")
        print(f"    Zero count   : {stats['zero_count']}")
        print(f"    Non-zero     : {stats['nonzero']}")
        if "min" in stats:
            print(f"    Min / Max    : {stats['min']:.4f} / {stats['max']:.4f}")
            print(f"    Mean ± Std   : {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    P25/P50/P75  : {stats['p25']:.4f} / {stats['p50']:.4f} / {stats['p75']:.4f}")

    # ── 2. Compute network_score for every cell ───────────────────────────────
    print("\n" + "=" * 60)
    print("NETWORK SCORE SIMULATION (per cell)")
    print("=" * 60)

    nq = NetworkQualityCalculator(config)
    scores       = []
    confidences  = []
    zero_scores  = 0
    fallback_used = 0

    for _, row in df.iterrows():
        cell = row.to_dict()
        info = nq.calculate(cell)
        s    = info["network_score"]
        scores.append(s)
        confidences.append(info["confidence"])
        if s == 0.0:
            zero_scores += 1
        if not info["available_metrics"] or \
           all(m in ("network_data_confidence", "nearest_tower_distance")
               for m in info["available_metrics"]):
            fallback_used += 1

    scores_arr = np.array(scores)
    print(f"\n  Cells with score = 0.0 : {zero_scores} / {len(df)}")
    print(f"  Fallback path used     : {fallback_used} / {len(df)}")
    print(f"  Score range            : [{scores_arr.min():.4f}, {scores_arr.max():.4f}]")
    print(f"  Score mean ± std       : {scores_arr.mean():.4f} ± {scores_arr.std():.4f}")

    if scores_arr.std() < 0.01:
        print("\n  ⚠️  WARNING: network_score has very low variance (<0.01).")
        print("     PPO cannot learn network-aware navigation with this data.")
        print("     → Add OpenCelliD API key and re-run the data pipeline.")
    else:
        print("\n  ✅ network_score has meaningful variation — network reward will work.")

    # ── 3. Tower source breakdown ─────────────────────────────────────────────
    if "network_source" in df.columns:
        print("\n" + "=" * 60)
        print("NETWORK SOURCE BREAKDOWN")
        print("=" * 60)
        src_counts = df["network_source"].value_counts(dropna=False)
        for src, count in src_counts.items():
            print(f"  {str(src):30s}: {count:5d}  ({100*count/len(df):.1f}%)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_network_diagnostics()
