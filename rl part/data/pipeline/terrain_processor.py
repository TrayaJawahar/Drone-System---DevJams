"""
terrain_processor.py
--------------------
Extracts elevation and slope values for each grid cell
from a downloaded DEM (Digital Elevation Model) GeoTIFF.

Uses rasterio for raster sampling and numpy for gradient computation.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from rl.utils.logger import setup_logger

logger = setup_logger("terrain_processor")


def extract_terrain_for_grid(grid_df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """
    Samples the DEM GeoTIFF at each grid cell center coordinate.
    Computes per-cell elevation and slope (gradient magnitude).

    Args:
        grid_df: DataFrame with latitude, longitude columns
        dem_path: Path to the DEM GeoTIFF file

    Returns:
        Updated grid_df with added columns:
            elevation (float, meters above sea level)
            slope (float, degrees, 0 = flat)
    """
    logger.info(f"Opening DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        logger.info(f"DEM CRS: {src.crs}")
        logger.info(f"DEM shape: {src.shape}, resolution: {src.res}")
        logger.info(f"DEM bounds: {src.bounds}")

        # Read the full elevation band
        dem_array = src.read(1).astype(float)

        # Replace nodata values with NaN
        nodata = src.nodata
        if nodata is not None:
            dem_array[dem_array == nodata] = np.nan

        transform = src.transform

        # Compute slope from the DEM array (gradient magnitude in degrees)
        slope_array = _compute_slope(dem_array, src.res)

        # Sample elevation and slope at each grid cell center
        n = len(grid_df)
        elevations = np.full(n, np.nan)
        slopes = np.full(n, 0.0)

        for i, row in grid_df.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]

            # Convert lat/lon to raster row/col index
            try:
                # Transform EPSG:4326 to raster CRS
                xs, ys = warp_transform('EPSG:4326', src.crs, [lon], [lat])
                rr, cc = rowcol(transform, xs[0], ys[0])
                
                # Clip to raster bounds to avoid off-by-one Edge NaNs
                rr = max(0, min(rr, dem_array.shape[0] - 1))
                cc = max(0, min(cc, dem_array.shape[1] - 1))
                
                elevations[i] = dem_array[rr, cc]
                slopes[i] = slope_array[rr, cc]
            except Exception as e:
                logger.debug(f"Could not sample DEM at ({lat}, {lon}): {e}")

            if (i + 1) % 500 == 0:
                logger.info(f"  Sampled {i + 1}/{n} cells...")

    # Fallback: interpolate missing elevations using mean
    valid_elevations = elevations[~np.isnan(elevations)]
    if len(valid_elevations) == 0:
        logger.warning("No valid elevation values found in DEM for this region. Defaulting to 0.")
        elevations = np.zeros(n)
    else:
        mean_elev = float(np.mean(valid_elevations))
        nan_count = int(np.isnan(elevations).sum())
        if nan_count > 0:
            logger.warning(f"{nan_count} cells have no DEM coverage; filling with mean elevation {mean_elev:.1f}m")
            elevations = np.where(np.isnan(elevations), mean_elev, elevations)

    grid_df = grid_df.copy()
    grid_df["elevation"] = np.round(elevations, 2)
    grid_df["slope"] = np.round(slopes, 3)

    logger.info(f"Terrain extraction complete. Elevation range: [{elevations.min():.1f}, {elevations.max():.1f}]m")
    logger.info(f"Slope range: [{slopes.min():.1f}, {slopes.max():.1f}]°")

    return grid_df


def _compute_slope(dem_array: np.ndarray, resolution: tuple) -> np.ndarray:
    """
    Computes terrain slope in degrees from a 2D elevation array.
    Uses central difference gradient.

    Args:
        dem_array: 2D numpy array of elevation values
        resolution: (x_res, y_res) in CRS units (degrees for WGS84)

    Returns:
        2D numpy array of slope values in degrees
    """
    # Convert resolution from degrees to meters for gradient calculation
    # 1 degree latitude ≈ 111320m
    y_res_m = abs(resolution[1]) * 111320.0
    x_res_m = abs(resolution[0]) * 111320.0

    # Gradient computation (forward difference on edges, central on interior)
    # nan-safe: use np.gradient which handles boundaries
    filled = np.where(np.isnan(dem_array), 0.0, dem_array)
    dz_dy, dz_dx = np.gradient(filled, y_res_m, x_res_m)

    # Gradient magnitude → slope in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    return slope_deg
