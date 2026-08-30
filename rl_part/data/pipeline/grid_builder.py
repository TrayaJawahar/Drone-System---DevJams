"""
grid_builder.py
---------------
Converts a geographic bounding box into an NxM grid of cells.
Each cell has: cell_id, grid_x, grid_y, latitude, longitude (cell center).
Also provides coordinate conversion utilities (GPS ↔ grid).
"""

import math
import numpy as np
import pandas as pd
from pyproj import Transformer


def compute_bounding_box(center_lat: float, center_lon: float,
                         width_km: float, height_km: float) -> dict:
    """
    Computes the bounding box corners from a center point and dimensions.
    Uses WGS84 → local metric conversion for accuracy.

    Returns:
        dict with keys: min_lat, max_lat, min_lon, max_lon
    """
    # 1 degree latitude ≈ 111.32 km everywhere
    # 1 degree longitude ≈ 111.32 * cos(lat) km
    half_h_deg = (height_km / 2.0) / 111.32
    half_w_deg = (width_km / 2.0) / (111.32 * math.cos(math.radians(center_lat)))

    return {
        "min_lat": center_lat - half_h_deg,
        "max_lat": center_lat + half_h_deg,
        "min_lon": center_lon - half_w_deg,
        "max_lon": center_lon + half_w_deg,
    }


def build_grid(bbox: dict, cell_size_meters: float) -> pd.DataFrame:
    """
    Creates a uniform grid of cells covering the bounding box.

    Args:
        bbox: dict with min_lat, max_lat, min_lon, max_lon
        cell_size_meters: resolution of each grid cell in meters

    Returns:
        pd.DataFrame with columns:
            cell_id, grid_x, grid_y, latitude, longitude
    """
    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]

    # Convert cell size to degrees
    cell_lat_deg = cell_size_meters / 111320.0
    cell_lon_deg = cell_size_meters / (111320.0 * math.cos(math.radians((min_lat + max_lat) / 2)))

    # Build lat/lon arrays for cell centers (offset by half cell)
    lats = np.arange(min_lat + cell_lat_deg / 2, max_lat, cell_lat_deg)
    lons = np.arange(min_lon + cell_lon_deg / 2, max_lon, cell_lon_deg)

    grid_height = len(lats)
    grid_width = len(lons)

    records = []
    cell_id = 0
    for gx, lon in enumerate(lons):
        for gy, lat in enumerate(lats):
            records.append({
                "cell_id": cell_id,
                "grid_x": gx,
                "grid_y": gy,
                "latitude": round(float(lat), 7),
                "longitude": round(float(lon), 7),
            })
            cell_id += 1

    df = pd.DataFrame(records)
    return df, grid_width, grid_height


def latlon_to_grid(lat: float, lon: float, bbox: dict,
                   grid_width: int, grid_height: int) -> tuple[int, int]:
    """
    Converts GPS coordinates to (grid_x, grid_y).
    Returns (-1, -1) if outside the grid.
    """
    lat_range = bbox["max_lat"] - bbox["min_lat"]
    lon_range = bbox["max_lon"] - bbox["min_lon"]

    if not (bbox["min_lat"] <= lat <= bbox["max_lat"] and
            bbox["min_lon"] <= lon <= bbox["max_lon"]):
        return -1, -1

    gx = int((lon - bbox["min_lon"]) / lon_range * grid_width)
    gy = int((lat - bbox["min_lat"]) / lat_range * grid_height)

    gx = min(gx, grid_width - 1)
    gy = min(gy, grid_height - 1)
    return gx, gy


def grid_to_latlon(gx: int, gy: int, bbox: dict,
                   grid_width: int, grid_height: int) -> tuple[float, float]:
    """
    Converts (grid_x, grid_y) to (latitude, longitude) of the cell center.
    """
    lat_range = bbox["max_lat"] - bbox["min_lat"]
    lon_range = bbox["max_lon"] - bbox["min_lon"]

    cell_lat = bbox["min_lat"] + (gy + 0.5) / grid_height * lat_range
    cell_lon = bbox["min_lon"] + (gx + 0.5) / grid_width * lon_range
    return round(cell_lat, 7), round(cell_lon, 7)
