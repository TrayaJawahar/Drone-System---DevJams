"""
obstacle_processor.py
---------------------
Processes raw OSM features to determine which grid cells are obstacles.

For each grid cell center (lat, lon), checks if it intersects with:
  - Building polygons → is_obstacle = True
  - Water bodies → is_obstacle = True
  - Roads → NOT obstacles (drones fly over roads)

Also computes:
  - obstacle_distance: distance in meters to nearest obstacle boundary

Uses Shapely + STRtree for efficient spatial indexing over large polygon sets.
"""

import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
from shapely import STRtree
from rl.utils.logger import setup_logger

logger = setup_logger("obstacle_processor")


def _is_obstacle_feature(tags: dict, config: dict) -> bool:
    """
    Returns True if the OSM tags represent an obstacle feature (building/water).
    Roads are explicitly excluded — drones fly over them.
    """
    osm_cfg = config.get("osm", {}) if config else {}
    tag_cfg = osm_cfg.get("obstacle_tags", {})

    # Explicit road → never an obstacle
    if "highway" in tags:
        return False

    # Buildings
    if "building" in tags and tag_cfg.get("buildings", True):
        return True

    # Water
    if tag_cfg.get("water", True):
        if tags.get("natural") == "water":
            return True
        if tags.get("landuse") in ("reservoir", "basin"):
            return True
        if tags.get("waterway") in ("river", "canal", "stream"):
            return True

    # Industrial (optional)
    if tags.get("landuse") == "industrial" and tag_cfg.get("industrial", False):
        return True

    return False


def _build_polygon(coords: list) -> Polygon | None:
    """Safely builds and validates a Shapely Polygon."""
    if len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            return poly
        # Try to fix invalid polygon
        fixed = poly.buffer(0)
        if not fixed.is_empty:
            return fixed
    except Exception:
        pass
    return None


def extract_osm_polygons(osm_data: dict, config: dict = None) -> tuple[list, list]:
    """
    Extracts obstacle polygons and road linestrings from Overpass API response.

    Returns:
        obstacle_polys: list of Shapely geometries (buildings + water bodies)
        road_lines: list of Shapely LineString objects
    """
    elements = osm_data.get("elements", [])
    logger.info(f"Processing {len(elements)} OSM elements...")

    obstacle_polys = []
    road_lines = []

    # Build node lookup dict for way assembly (when geometry is not inline)
    node_lookup = {}
    for el in elements:
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            node_lookup[el["id"]] = (el["lon"], el["lat"])

    for el in elements:
        el_type = el.get("type")
        tags = el.get("tags", {})

        if el_type == "way":
            # Get geometry — Overpass `out geom` provides inline geometry
            if "geometry" in el:
                coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
            elif "nodes" in el:
                coords = [node_lookup[n] for n in el["nodes"] if n in node_lookup]
            else:
                continue

            is_road = "highway" in tags

            if is_road:
                # Road → not obstacle, store as linestring for reference
                if len(coords) >= 2:
                    try:
                        road_lines.append(LineString(coords))
                    except Exception:
                        pass
            else:
                # Potential obstacle polygon
                if _is_obstacle_feature(tags, config):
                    poly = _build_polygon(coords)
                    if poly is not None:
                        obstacle_polys.append(poly)

        elif el_type == "relation":
            tags = el.get("tags", {})
            if not _is_obstacle_feature(tags, config):
                continue

            # Multipolygon relation
            members = el.get("members", [])
            outer_rings = []
            for member in members:
                if member.get("role") == "outer" and "geometry" in member:
                    coords = [(g["lon"], g["lat"]) for g in member["geometry"]]
                    poly = _build_polygon(coords)
                    if poly is not None:
                        outer_rings.append(poly)

            if outer_rings:
                try:
                    if len(outer_rings) == 1:
                        obstacle_polys.append(outer_rings[0])
                    else:
                        merged = unary_union(outer_rings)
                        if merged and not merged.is_empty:
                            obstacle_polys.append(merged)
                except Exception:
                    obstacle_polys.extend(outer_rings)

    logger.info(f"Extracted {len(obstacle_polys)} obstacle polygons, {len(road_lines)} road segments")
    return obstacle_polys, road_lines


def assign_obstacles_to_grid(grid_df: pd.DataFrame,
                              obstacle_polys: list,
                              road_lines: list,
                              config: dict = None) -> pd.DataFrame:
    """
    Uses Shapely STRtree spatial index to efficiently determine which grid cells
    are inside obstacle polygons, and compute distance to nearest obstacle.

    Args:
        grid_df: DataFrame with latitude, longitude columns
        obstacle_polys: list of Shapely Polygon objects
        road_lines: list of Shapely LineString objects (reference only)
        config: pipeline config (unused here, for API parity)

    Returns:
        Updated grid_df with is_obstacle (bool) and obstacle_distance (float, meters).
    """
    grid_df = grid_df.copy()

    if not obstacle_polys:
        logger.warning("No obstacle polygons found. All cells will be non-obstacle.")
        grid_df["is_obstacle"] = False
        grid_df["obstacle_distance"] = 9999.0
        return grid_df

    logger.info(f"Building STRtree spatial index for {len(obstacle_polys)} obstacle polygons...")
    tree = STRtree(obstacle_polys)

    n = len(grid_df)
    is_obstacle_arr = np.zeros(n, dtype=bool)
    obstacle_dist_arr = np.full(n, 9999.0, dtype=float)

    # Build Shapely Point objects for all grid cells
    points = [Point(row["longitude"], row["latitude"]) for _, row in grid_df.iterrows()]

    logger.info(f"Testing {n} grid cells against obstacle polygons...")
    for i, pt in enumerate(points):
        # Query nearby polygons via STRtree
        candidate_idxs = tree.query(pt)

        min_dist_deg = float("inf")
        inside = False

        for cidx in candidate_idxs:
            poly = obstacle_polys[cidx]
            if poly.contains(pt):
                inside = True
                min_dist_deg = 0.0
                break
            else:
                d = poly.distance(pt)
                if d < min_dist_deg:
                    min_dist_deg = d

        is_obstacle_arr[i] = inside

        # Convert degree distance to meters (approx)
        if min_dist_deg == float("inf"):
            # No nearby candidates at all → find global nearest
            # Use a coarser approximation: just set large distance
            obstacle_dist_arr[i] = 9999.0
        else:
            obstacle_dist_arr[i] = max(0.0, min_dist_deg * 111320.0)

        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{n} cells...")

    grid_df["is_obstacle"] = is_obstacle_arr
    grid_df["obstacle_distance"] = np.round(obstacle_dist_arr, 1)

    obstacle_count = int(is_obstacle_arr.sum())
    logger.info(
        f"Obstacle assignment complete: "
        f"{obstacle_count}/{n} cells are obstacles ({100 * obstacle_count / n:.1f}%)"
    )

    return grid_df
