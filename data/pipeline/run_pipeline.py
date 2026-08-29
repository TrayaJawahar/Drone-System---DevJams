"""
run_pipeline.py
---------------
Master pipeline runner for building the Geo-Network Map.

Executes all stages in order:
  1. Load configuration
  2. Compute bounding box from region settings
  3. Build base grid (lat/lon cells)
  4. Download OSM features (Overpass API)
  5. Download terrain DEM (OpenTopography API)
  6. Download cell towers (OpenCelliD API)
  7. Process obstacles from OSM
  8. Extract elevation + slope from DEM
  9. Estimate network quality from tower positions
  10. Assemble and save final parquet + metadata

Usage:
  python data/pipeline/run_pipeline.py

Configuration:
  Edit data/config/pipeline_config.yaml to change region or API settings.
  Set API keys in data/config/pipeline_config.yaml before running.
"""

import os
import sys
import json
import time
import yaml
import argparse

# Add project root to path so rl.utils.logger is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rl.utils.logger import setup_logger
from data.pipeline.grid_builder import compute_bounding_box, build_grid
from data.pipeline.osm_collector import download_osm_data, load_osm_data
from data.pipeline.terrain_collector import download_terrain_dem
from data.pipeline.tower_collector import download_towers_in_area, load_towers
from data.pipeline.obstacle_processor import extract_osm_polygons, assign_obstacles_to_grid
from data.pipeline.terrain_processor import extract_terrain_for_grid
from data.pipeline.network_estimator import estimate_network_for_grid
from data.pipeline.map_assembler import assemble_geo_network_map

logger = setup_logger("pipeline")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.yaml")
APIKEYS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "api_keys.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Inject API keys from api_keys.yaml if it exists
    keys_path = APIKEYS_PATH
    if os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as f:
            keys = yaml.safe_load(f) or {}
        topo_key = keys.get("opentopography_api_key", "")
        cell_key = keys.get("opencellid_api_key", "")
        if topo_key and not topo_key.startswith("YOUR_"):
            cfg.setdefault("terrain", {})["api_key"] = topo_key
        if cell_key and not cell_key.startswith("YOUR_"):
            cfg.setdefault("towers", {})["api_key"] = cell_key
    else:
        logger.warning(f"api_keys.yaml not found at {keys_path}. API keys must be set in pipeline_config.yaml.")

    return cfg


def run_pipeline(config_path: str = CONFIG_PATH,
                 skip_download: bool = False,
                 steps: list = None) -> str:
    """
    Runs the full data preparation pipeline.

    Args:
        config_path: Path to pipeline_config.yaml
        skip_download: If True, skip API calls and use cached raw files
        steps: Optional list of specific steps to run (for debugging)

    Returns:
        Path to the final geo_network_map.parquet
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("DRONE GEO-NETWORK MAP PIPELINE STARTING")
    logger.info("=" * 60)

    # --- Step 0: Load configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    region_cfg = config["region"]
    grid_cfg = config["grid"]
    output_cfg = config["output"]

    region_name = region_cfg["name"]
    center_lat = region_cfg["center_latitude"]
    center_lon = region_cfg["center_longitude"]
    width_km = region_cfg["width_km"]
    height_km = region_cfg["height_km"]
    cell_size_m = grid_cfg["cell_size_meters"]

    output_parquet = os.path.join(PROJECT_ROOT, output_cfg["parquet_path"])
    output_metadata = os.path.join(PROJECT_ROOT, output_cfg["metadata_path"])
    raw_osm_dir = os.path.join(PROJECT_ROOT, output_cfg["raw_osm_path"])
    raw_terrain_dir = os.path.join(PROJECT_ROOT, output_cfg["raw_terrain_path"])
    raw_towers_dir = os.path.join(PROJECT_ROOT, output_cfg["raw_towers_path"])

    logger.info(f"Target region: {region_name}")
    logger.info(f"Center: ({center_lat}, {center_lon})")
    logger.info(f"Area: {width_km}km x {height_km}km")
    logger.info(f"Cell size: {cell_size_m}m")

    # --- Step 1: Compute bounding box ---
    logger.info("\n[Step 1/9] Computing geographic bounding box...")
    bbox = compute_bounding_box(center_lat, center_lon, width_km, height_km)
    logger.info(f"Bounding box: {bbox}")

    # --- Step 2: Build base grid ---
    logger.info(f"\n[Step 2/9] Building {cell_size_m}m resolution grid...")
    grid_df, grid_width, grid_height = build_grid(bbox, cell_size_m)
    total_cells = len(grid_df)
    logger.info(f"Grid: {grid_width} x {grid_height} = {total_cells:,} cells")

    # --- Step 3: Download OSM data ---
    logger.info("\n[Step 3/9] Downloading OSM geographic data (Overpass API)...")
    if not skip_download:
        osm_path = download_osm_data(bbox, config, raw_osm_dir)
    else:
        osm_path = os.path.join(raw_osm_dir, "osm_features.json")
        if not os.path.exists(osm_path):
            raise FileNotFoundError(f"skip_download=True but cached OSM file not found: {osm_path}")
        logger.info(f"Using cached OSM data: {osm_path}")

    # --- Step 4: Download terrain DEM ---
    logger.info("\n[Step 4/9] Downloading terrain DEM (OpenTopography API)...")
    dem_path = None
    terrain_available = True
    if not skip_download:
        try:
            dem_path = download_terrain_dem(bbox, config, raw_terrain_dir)
        except ValueError as e:
            logger.warning(f"Terrain download skipped: {e}")
            logger.warning("Elevation and slope will not be available in the map.")
            terrain_available = False
    else:
        dem_path = os.path.join(raw_terrain_dir, "dem.tif")
        terrain_available = os.path.exists(dem_path)
        if not terrain_available:
            logger.warning("Cached DEM not found. Terrain data will be skipped.")

    # --- Step 5: Download tower data ---
    logger.info("\n[Step 5/9] Downloading cell tower data (OpenCelliD API)...")
    towers_path = None
    towers_available = True
    if not skip_download:
        try:
            towers_path = download_towers_in_area(bbox, config, raw_towers_dir)
        except ValueError as e:
            logger.warning(f"Tower download skipped: {e}")
            logger.warning("Network quality metrics will not be available.")
            towers_available = False
    else:
        towers_path = os.path.join(raw_towers_dir, "towers.csv")
        towers_available = os.path.exists(towers_path)
        if not towers_available:
            logger.warning("Cached towers CSV not found. Network data will be skipped.")

    # --- Step 6: Process obstacles ---
    logger.info("\n[Step 6/9] Processing OSM obstacles...")
    osm_data = load_osm_data(osm_path)
    obstacle_polys, road_lines = extract_osm_polygons(osm_data, config)
    grid_df = assign_obstacles_to_grid(grid_df, obstacle_polys, road_lines, config)
    obstacle_pct = 100.0 * grid_df["is_obstacle"].sum() / total_cells
    logger.info(f"Obstacles: {grid_df['is_obstacle'].sum():,}/{total_cells:,} cells ({obstacle_pct:.1f}%)")

    # --- Step 7: Extract terrain ---
    if terrain_available and dem_path:
        logger.info("\n[Step 7/9] Extracting terrain from DEM...")
        grid_df = extract_terrain_for_grid(grid_df, dem_path)
    else:
        logger.info("\n[Step 7/9] Terrain extraction SKIPPED (no DEM available)")
        grid_df["elevation"] = None
        grid_df["slope"] = None

    # --- Step 8: Estimate network quality ---
    if towers_available and towers_path:
        logger.info("\n[Step 8/9] Estimating network quality from tower data...")
        towers = load_towers(towers_path)
        grid_df = estimate_network_for_grid(grid_df, towers, config)
    else:
        logger.info("\n[Step 8/9] Network estimation SKIPPED (no tower data)")
        for col in ["nearest_tower_distance", "rssi", "rsrp", "sinr",
                    "throughput", "latency", "packet_loss",
                    "network_data_confidence", "tower_count_nearby"]:
            grid_df[col] = None
        grid_df["network_source"] = "missing"

    # --- Step 9: Assemble and save final map ---
    logger.info("\n[Step 9/9] Assembling and saving Geo-Network Map...")
    final_df = assemble_geo_network_map(
        grid_df=grid_df,
        bbox=bbox,
        grid_width=grid_width,
        grid_height=grid_height,
        cell_size_meters=cell_size_m,
        config=config,
        output_parquet=output_parquet,
        output_metadata=output_metadata,
    )

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline completed in {elapsed:.1f}s")
    logger.info(f"Output: {output_parquet}")
    logger.info(f"Metadata: {output_metadata}")
    logger.info("\nYou can now validate and start training:")
    logger.info("  python -m rl.data.validation")
    logger.info("  python run_training.py")

    return output_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Geo-Network Map from real API data."
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to pipeline_config.yaml (default: data/config/pipeline_config.yaml)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip API downloads and use cached raw files in data/raw/"
    )
    args = parser.parse_args()

    try:
        output_path = run_pipeline(
            config_path=args.config,
            skip_download=args.skip_download,
        )
        print(f"\nMap ready: {output_path}")
        sys.exit(0)
    except ValueError as e:
        # API key not configured or similar user-error
        print(f"\n[CONFIG ERROR] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[FILE ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[API ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
