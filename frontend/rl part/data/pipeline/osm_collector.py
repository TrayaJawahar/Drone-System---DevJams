"""
osm_collector.py
----------------
Downloads real geographic and obstacle data from the Overpass API
(OpenStreetMap). No API key required.

Collects:
  - Building footprints → obstacles
  - Water bodies (rivers, lakes) → obstacles
  - Roads → navigable corridors (non-obstacles)

Saves raw GeoJSON to data/raw/osm/.
"""

import os
import json
import time
import requests
from rl.utils.logger import setup_logger

logger = setup_logger("osm_collector")

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"


def build_overpass_query(bbox: dict, config: dict) -> str:
    """
    Builds an Overpass QL query to extract OSM features within the bounding box.
    Returns two sections: obstacles (buildings, water) and roads.
    """
    min_lat = bbox["min_lat"]
    min_lon = bbox["min_lon"]
    max_lat = bbox["max_lat"]
    max_lon = bbox["max_lon"]
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    osm_cfg = config.get("osm", {})
    tags = osm_cfg.get("obstacle_tags", {})

    queries = ['[out:json][timeout:180];', '(']

    # --- OBSTACLES ---
    # Buildings
    if tags.get("buildings", True):
        queries.append(f'  way["building"]({bbox_str});')
        queries.append(f'  relation["building"]({bbox_str});')

    # Water bodies
    if tags.get("water", True):
        queries.append(f'  way["natural"="water"]({bbox_str});')
        queries.append(f'  relation["natural"="water"]({bbox_str});')
        queries.append(f'  way["landuse"="reservoir"]({bbox_str});')
        queries.append(f'  way["waterway"="river"]({bbox_str});')
        queries.append(f'  way["waterway"="canal"]({bbox_str});')

    # Industrial (optional)
    if tags.get("industrial", False):
        queries.append(f'  way["landuse"="industrial"]({bbox_str});')

    # --- ROADS (fetched separately, tagged with type=road for identification) ---
    # Roads are NOT obstacles; they are collected only to detect navigable corridors.
    queries.append(f'  way["highway"]["highway"!="footway"]["highway"!="steps"]({bbox_str});')

    queries.append(');')
    queries.append('out body geom;')

    return '\n'.join(queries)


def download_osm_data(bbox: dict, config: dict, output_dir: str) -> str:
    """
    Downloads OSM obstacle features using the Overpass API.

    Returns:
        Path to the saved JSON file.

    Raises:
        RuntimeError: If the API call fails after max_retries.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "osm_features.json")

    # Check cache
    if os.path.exists(output_path):
        logger.info(f"Using cached OSM data at {output_path}")
        return output_path

    osm_cfg = config.get("osm", {})
    endpoint = osm_cfg.get("endpoint", OVERPASS_ENDPOINT)
    timeout = osm_cfg.get("timeout", 120)
    max_retries = osm_cfg.get("max_retries", 3)

    query = build_overpass_query(bbox, config)
    logger.info(f"Sending Overpass API query for bbox: {bbox}")
    logger.debug(f"Query:\n{query}")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout,
                headers={"User-Agent": "DroneRoutePlanning/1.0 (research project)"}
            )
            response.raise_for_status()
            data = response.json()

            element_count = len(data.get("elements", []))
            logger.info(f"OSM API returned {element_count} elements")

            if element_count == 0:
                logger.warning("Overpass API returned 0 elements. Check your bounding box or query.")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            logger.info(f"Saved OSM data to {output_path}")
            return output_path

        except requests.exceptions.Timeout:
            last_error = f"Attempt {attempt}/{max_retries}: Overpass API request timed out."
            logger.warning(last_error)
        except requests.exceptions.RequestException as e:
            last_error = f"Attempt {attempt}/{max_retries}: Overpass API error: {e}"
            logger.warning(last_error)
        except (json.JSONDecodeError, KeyError) as e:
            last_error = f"Attempt {attempt}/{max_retries}: Failed to parse Overpass response: {e}"
            logger.warning(last_error)

        if attempt < max_retries:
            wait = 10 * attempt
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Overpass API failed after {max_retries} attempts. Last error: {last_error}")


def load_osm_data(osm_path: str) -> dict:
    """
    Loads OSM data from the cached JSON file.
    """
    with open(osm_path, "r", encoding="utf-8") as f:
        return json.load(f)
