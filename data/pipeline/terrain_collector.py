"""
terrain_collector.py
--------------------
Downloads real Digital Elevation Model (DEM) data from the OpenTopography API.
Provides elevation and terrain height for each grid cell.

API docs: https://opentopography.org/developers
Requires an API key (free, register at opentopography.org).
"""

import os
import time
import requests
from rl.utils.logger import setup_logger

logger = setup_logger("terrain_collector")

OPENTOPO_ENDPOINT = "https://portal.opentopography.org/API/globaldem"


def download_terrain_dem(bbox: dict, config: dict, output_dir: str) -> str:
    """
    Downloads a DEM (Digital Elevation Model) GeoTIFF from OpenTopography.

    Args:
        bbox: dict with min_lat, max_lat, min_lon, max_lon
        config: pipeline config dict
        output_dir: directory to save the downloaded GeoTIFF

    Returns:
        Path to the saved GeoTIFF file.

    Raises:
        ValueError: If the API key is not configured.
        RuntimeError: If the download fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dem.tif")

    # Check cache
    if os.path.exists(output_path):
        logger.info(f"Using cached DEM at {output_path}")
        return output_path

    terrain_cfg = config.get("terrain", {})
    api_key = terrain_cfg.get("api_key", "")

    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "OpenTopography API key not configured.\n"
            "Please set terrain.api_key in data/config/pipeline_config.yaml.\n"
            "Get a free key at: https://portal.opentopography.org/requestApiKey"
        )

    endpoint = terrain_cfg.get("endpoint", OPENTOPO_ENDPOINT)
    dem_type = terrain_cfg.get("dem_type", "SRTMGL1")

    params = {
        "demtype": dem_type,
        "south": bbox["min_lat"],
        "north": bbox["max_lat"],
        "west": bbox["min_lon"],
        "east": bbox["max_lon"],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    logger.info(f"Requesting DEM ({dem_type}) from OpenTopography for bbox: {bbox}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=300,  # DEM downloads can take time
                stream=True,
                headers={"User-Agent": "DroneRoutePlanning/1.0 (research project)"}
            )

            if response.status_code == 401:
                raise ValueError(
                    "OpenTopography API key is invalid or expired. "
                    "Please check your key at https://portal.opentopography.org"
                )

            if response.status_code == 429:
                wait = 60 * attempt
                logger.warning(f"Rate limited by OpenTopography. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            # Check content type — should be image/tiff
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type or "text" in content_type:
                # Error returned as JSON/text
                error_text = response.text[:500]
                raise RuntimeError(f"OpenTopography returned error response: {error_text}")

            # Stream save the GeoTIFF
            total_bytes = 0
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)

            size_mb = total_bytes / (1024 * 1024)
            logger.info(f"DEM downloaded: {output_path} ({size_mb:.2f} MB)")
            return output_path

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Attempt {attempt}/{max_retries}: Network error: {e}")
        except ValueError:
            raise  # Re-raise config errors immediately
        except RuntimeError:
            raise

        if attempt < max_retries:
            wait = 15 * attempt
            logger.info(f"Retrying DEM download in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"OpenTopography DEM download failed after {max_retries} attempts.")
