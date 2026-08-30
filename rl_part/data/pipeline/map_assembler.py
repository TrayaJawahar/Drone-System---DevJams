"""
map_assembler.py
----------------
Assembles the final Geo-Network Map from all processed layers:
  - Grid cells (lat/lon, grid_x, grid_y)
  - Obstacle flags from OSM
  - Terrain (elevation, slope)
  - Network estimates (rssi, rsrp, sinr, throughput)

Outputs:
  - data/processed/geo_network_map.parquet
  - data/processed/geo_network_metadata.json

This is the final step in the data preparation pipeline.
The RL training system reads directly from these files.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from rl.utils.logger import setup_logger

logger = setup_logger("map_assembler")

# Columns required by the RL training system (from data_loader.py)
REQUIRED_COLUMNS = ["cell_id", "grid_x", "grid_y", "latitude", "longitude", "is_obstacle"]

# All possible network metric columns
NETWORK_COLUMNS = ["rssi", "rsrp", "sinr", "latency", "packet_loss", "throughput"]


def assemble_geo_network_map(grid_df: pd.DataFrame,
                              bbox: dict,
                              grid_width: int,
                              grid_height: int,
                              cell_size_meters: float,
                              config: dict,
                              output_parquet: str,
                              output_metadata: str) -> pd.DataFrame:
    """
    Merges all processed layers into the final geo_network_map DataFrame.
    Validates required columns, writes parquet and metadata JSON.

    Args:
        grid_df: DataFrame with all processed columns merged in
        bbox: Geographic bounding box dict
        grid_width: Number of columns in the grid
        grid_height: Number of rows in the grid
        cell_size_meters: Resolution of each grid cell
        config: Pipeline configuration dict
        output_parquet: Path to write the .parquet file
        output_metadata: Path to write the .json metadata file

    Returns:
        Final assembled DataFrame
    """
    logger.info("Assembling final Geo-Network Map...")

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in grid_df.columns]
    if missing:
        raise ValueError(f"Missing required columns after processing: {missing}")

    # Ensure is_obstacle is boolean
    grid_df["is_obstacle"] = grid_df["is_obstacle"].astype(bool)

    # Column ordering: required first, then optional network, then rest
    optional_cols = [c for c in NETWORK_COLUMNS if c in grid_df.columns]
    extra_cols = [c for c in grid_df.columns
                  if c not in REQUIRED_COLUMNS and c not in optional_cols]
    ordered_cols = REQUIRED_COLUMNS + optional_cols + extra_cols
    grid_df = grid_df[ordered_cols]

    # Final validation
    assert grid_df["cell_id"].is_unique, "cell_id must be unique"
    assert not grid_df[["grid_x", "grid_y"]].duplicated().any(), "Grid coordinates must be unique"

    # Write parquet
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    grid_df.to_parquet(output_parquet, index=False)
    file_size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    logger.info(f"Saved geo_network_map.parquet: {output_parquet} ({file_size_mb:.2f} MB)")

    # Build metadata
    region_cfg = config.get("region", {})
    total_cells = len(grid_df)
    obstacle_count = int(grid_df["is_obstacle"].sum())
    free_count = total_cells - obstacle_count

    # Network coverage stats
    network_coverage = {}
    for col in NETWORK_COLUMNS:
        if col in grid_df.columns:
            non_null = int(grid_df[col].notna().sum())
            network_coverage[col] = {
                "cells_with_data": non_null,
                "coverage_pct": round(100.0 * non_null / total_cells, 1),
                "min": round(float(grid_df[col].min()), 3) if non_null > 0 else None,
                "max": round(float(grid_df[col].max()), 3) if non_null > 0 else None,
                "mean": round(float(grid_df[col].mean()), 3) if non_null > 0 else None,
            }
        else:
            network_coverage[col] = {"cells_with_data": 0, "coverage_pct": 0.0}

    # Terrain stats
    terrain_stats = {}
    for col in ["elevation", "slope"]:
        if col in grid_df.columns:
            terrain_stats[col] = {
                "min": round(float(grid_df[col].min()), 2),
                "max": round(float(grid_df[col].max()), 2),
                "mean": round(float(grid_df[col].mean()), 2),
            }

    metadata = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "1.0.0",
        "region": {
            "name": region_cfg.get("name", "Unknown"),
            "center_latitude": region_cfg.get("center_latitude"),
            "center_longitude": region_cfg.get("center_longitude"),
            "width_km": region_cfg.get("width_km"),
            "height_km": region_cfg.get("height_km"),
        },
        "bounding_box": bbox,
        "grid": {
            "width": grid_width,
            "height": grid_height,
            "cell_size_meters": cell_size_meters,
            "total_cells": total_cells,
        },
        "map_stats": {
            "total_cells": total_cells,
            "obstacle_cells": obstacle_count,
            "free_cells": free_count,
            "obstacle_pct": round(100.0 * obstacle_count / total_cells, 1),
        },
        "network_coverage": network_coverage,
        "terrain": terrain_stats,
        "data_sources": {
            "geographic": "OpenStreetMap Overpass API",
            "terrain": "OpenTopography SRTM DEM",
            "towers": "OpenCelliD API",
            "network_quality": "ITU-R log-distance path-loss model (estimated from tower positions)",
            "latency": "not_available (requires drive-test measurements)",
            "packet_loss": "not_available (requires drive-test measurements)",
        },
        "coordinate_system": {
            "crs": "WGS84 (EPSG:4326)",
            "grid_origin": "bottom-left (min_lon, min_lat)",
            "grid_x_direction": "east (increasing longitude)",
            "grid_y_direction": "north (increasing latitude)",
        },
        "columns": list(grid_df.columns),
    }

    os.makedirs(os.path.dirname(output_metadata), exist_ok=True)
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved geo_network_metadata.json: {output_metadata}")

    # Print summary
    _print_summary(metadata)

    return grid_df


def _print_summary(metadata: dict):
    stats = metadata["map_stats"]
    net = metadata["network_coverage"]
    region = metadata["region"]

    print("\n" + "=" * 50)
    print("GEO-NETWORK MAP ASSEMBLED SUCCESSFULLY")
    print("=" * 50)
    print(f"Region: {region['name']}")
    print(f"Grid: {metadata['grid']['width']} x {metadata['grid']['height']} cells")
    print(f"Resolution: {metadata['grid']['cell_size_meters']}m per cell")
    print(f"Total Cells: {stats['total_cells']:,}")
    print(f"Obstacle Cells: {stats['obstacle_cells']:,} ({stats['obstacle_pct']}%)")
    print(f"Free Cells: {stats['free_cells']:,}")
    print()
    print("Network Coverage:")
    for metric, info in net.items():
        pct = info.get("coverage_pct", 0)
        mean = info.get("mean")
        mean_str = f"  mean={mean:.1f}" if mean is not None else ""
        print(f"  {metric.upper()}: {pct:.1f}%{mean_str}")
    print("=" * 50)
