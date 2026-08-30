import os
import json
import pandas as pd
from rl.utils.logger import setup_logger

logger = setup_logger("data_loader")

def load_geo_network_map(map_path: str, metadata_path: str):
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"ERROR: Real Geo-Network Map not found.\n"
            f"Please build {map_path} before training."
        )
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"ERROR: Geo-Network Metadata not found.\n"
            f"Please build {metadata_path} before training."
        )

    try:
        df = pd.read_parquet(map_path)
    except Exception as e:
        raise ValueError(f"Failed to read Parquet file at {map_path}: {e}")

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read JSON metadata at {metadata_path}: {e}")

    required_cols = ["cell_id", "grid_x", "grid_y", "latitude", "longitude", "is_obstacle"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in parquet map: {missing_cols}")
    if not df["cell_id"].is_unique:
        raise ValueError("Cell IDs are not unique in the Geo-Network Map Parquet.")

    # Check valid grid_x and grid_y
    if df["grid_x"].isnull().any() or df["grid_y"].isnull().any():
        raise ValueError("Grid coordinates (grid_x, grid_y) must not contain null values.")
    
    # Check uniqueness of grid coordinates
    duplicates = df[df.duplicated(subset=["grid_x", "grid_y"], keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Grid coordinates (grid_x, grid_y) are not unique. Duplicate positions found: {duplicates[['grid_x', 'grid_y']].head()}")

    # Check valid obstacle values
    if not df["is_obstacle"].dtype == bool and not pd.api.types.is_bool_dtype(df["is_obstacle"]):
        # Try converting to boolean
        try:
            df["is_obstacle"] = df["is_obstacle"].astype(bool)
        except Exception as e:
            raise ValueError(f"is_obstacle column is not boolean and could not be cast: {e}")

    return df, metadata

def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Computes a data quality summary from the Geo-Network Map dataframe.
    """
    total_cells = len(df)
    obstacle_cells = int(df["is_obstacle"].sum())
    free_cells = total_cells - obstacle_cells

    # Optional network metrics coverage
    network_metrics = ["rssi", "rsrp", "sinr", "latency", "packet_loss", "throughput"]
    coverage = {}
    for metric in network_metrics:
        if metric in df.columns:
            non_null_count = df[metric].notnull().sum()
            coverage[metric] = (non_null_count / total_cells) * 100.0
        else:
            coverage[metric] = 0.0

    # Source provenance
    provenance = {"measured": 0.0, "interpolated": 0.0, "missing": 0.0}
    if "network_source" in df.columns:
        source_counts = df["network_source"].value_counts(dropna=False)
        total_sources = source_counts.sum()
        
        measured_count = 0
        for idx in source_counts.index:
            if pd.isnull(idx) or str(idx).lower() in ["missing", "none", "nan", "null"]:
                continue
            if "measure" in str(idx).lower():
                measured_count += source_counts[idx]
        
        interpolated_count = 0
        for idx in source_counts.index:
            if pd.isnull(idx) or str(idx).lower() in ["missing", "none", "nan", "null"]:
                continue
            if "interpolate" in str(idx).lower() or "estimate" in str(idx).lower():
                interpolated_count += source_counts[idx]
        
        missing_count = total_cells - measured_count - interpolated_count
        
        provenance["measured"] = (measured_count / total_cells) * 100.0
        provenance["interpolated"] = (interpolated_count / total_cells) * 100.0
        provenance["missing"] = (missing_count / total_cells) * 100.0
    else:
        # Fallback based on RSSI non-null/null
        if "rssi" in df.columns:
            non_null_rssi = df["rssi"].notnull().sum()
            provenance["measured"] = (non_null_rssi / total_cells) * 100.0
            provenance["missing"] = ((total_cells - non_null_rssi) / total_cells) * 100.0
        else:
            provenance["missing"] = 100.0

    return {
        "total_cells": total_cells,
        "free_cells": free_cells,
        "obstacle_cells": obstacle_cells,
        "coverage": coverage,
        "provenance": provenance
    }

def print_quality_summary_report(summary: dict):
    """
    Prints a formatted data quality report.
    """
    report = []
    report.append("=" * 40)
    report.append("GEO-NETWORK MAP SUMMARY")
    report.append("=" * 40)
    report.append(f"Total Cells: {summary['total_cells']:,}")
    report.append(f"Free Cells: {summary['free_cells']:,}")
    report.append(f"Obstacle Cells: {summary['obstacle_cells']:,}")
    report.append("")
    report.append("Network Coverage:")
    for metric, cov in summary['coverage'].items():
        report.append(f"  {metric.upper()}: {cov:.1f}%")
    report.append("")
    report.append("Data Provenance:")
    report.append(f"  Measured Data: {summary['provenance']['measured']:.1f}%")
    report.append(f"  Interpolated Data: {summary['provenance']['interpolated']:.1f}%")
    report.append(f"  Missing Data: {summary['provenance']['missing']:.1f}%")
    report.append("=" * 40)
    
    report_str = "\n".join(report)
    print(report_str)
    return report_str
