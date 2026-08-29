"""
tower_collector.py
------------------
Downloads real cellular tower data from the OpenCelliD API.
Provides tower locations, radio type (GSM/LTE/NR), MCC, MNC.

API docs: https://wiki.opencellid.org/wiki/API
Requires an API key (free, register at opencellid.org).

Note: OpenCelliD supports two access modes:
  1. getInArea endpoint (small areas, limited rows)
  2. Bulk CSV download (recommended for large areas — downloaded as .csv.gz)

This collector uses getInArea for the 5km x 5km Koramangala region,
with automatic fallback guidance for bulk download if quota exceeded.
"""

import os
import time
import csv
import gzip
import json
import requests
from rl.utils.logger import setup_logger

logger = setup_logger("tower_collector")

OPENCELLID_ENDPOINT = "https://opencellid.org/cell/getInArea"


def download_towers_in_area(bbox: dict, config: dict, output_dir: str) -> str:
    """
    Downloads cell tower data for the bounding box from OpenCelliD.

    Args:
        bbox: dict with min_lat, max_lat, min_lon, max_lon
        config: pipeline config dict
        output_dir: directory to save the tower CSV

    Returns:
        Path to the saved towers CSV file.

    Raises:
        ValueError: If the API key is not configured.
        RuntimeError: If the download fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "towers.csv")

    # Check cache
    if os.path.exists(output_path):
        row_count = _count_csv_rows(output_path)
        logger.info(f"Using cached tower data at {output_path} ({row_count} towers)")
        return output_path

    tower_cfg = config.get("towers", {})
    api_key = tower_cfg.get("api_key", "")

    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "OpenCelliD API key not configured.\n"
            "Please set towers.api_key in data/config/pipeline_config.yaml.\n"
            "Get a free key at: https://opencellid.org/users/sign_in\n\n"
            "Alternatively, download the bulk CSV from:\n"
            "  https://opencellid.org/downloads (India: MCC 404, 405)\n"
            "And place it at: data/raw/towers/towers.csv"
        )

    endpoint = tower_cfg.get("endpoint", OPENCELLID_ENDPOINT)
    radio_filter = tower_cfg.get("radio_filter", [])
    mcc_filter = tower_cfg.get("mcc_filter", [])

    # OpenCelliD getInArea parameters
    params = {
        "key": api_key,
        "BBOX": f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}",
        "format": "csv",
    }

    # Add radio filter if specified
    if radio_filter:
        params["radio"] = ",".join(radio_filter)

    logger.info(f"Fetching cell towers from OpenCelliD for bbox: {bbox}")
    logger.info(f"Radio filter: {radio_filter if radio_filter else 'all'}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=120,
                headers={"User-Agent": "DroneRoutePlanning/1.0 (research project)"}
            )

            if response.status_code == 401 or response.status_code == 403:
                raise ValueError(
                    "OpenCelliD API key is invalid or access denied. "
                    "Please check your key at https://opencellid.org"
                )

            if response.status_code == 429:
                wait = 60 * attempt
                logger.warning(f"OpenCelliD rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            content = response.text
            if not content.strip():
                logger.warning("OpenCelliD returned empty response for this region.")
                # Write empty CSV with headers
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    f.write("radio,mcc,net,area,cell,unit,lon,lat,range,samples,changeable,created,updated,averageSignal\n")
                return output_path

            # Check for error response (sometimes returned as JSON)
            if content.strip().startswith("{"):
                error_data = json.loads(content)
                error_msg = error_data.get("message", str(error_data))
                raise RuntimeError(f"OpenCelliD API error: {error_msg}")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            row_count = _count_csv_rows(output_path)
            logger.info(f"Saved {row_count} towers to {output_path}")

            # Apply MCC filter post-download
            if mcc_filter:
                _filter_towers_by_mcc(output_path, mcc_filter)

            return output_path

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Attempt {attempt}/{max_retries}: Network error: {e}")
        except ValueError:
            raise  # Re-raise config errors immediately
        except RuntimeError:
            raise

        if attempt < max_retries:
            wait = 15 * attempt
            logger.info(f"Retrying tower download in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"OpenCelliD download failed after {max_retries} attempts.")


def load_towers_from_bulk_csv(csv_path: str, bbox: dict,
                               radio_filter: list = None, mcc_filter: list = None) -> str:
    """
    Loads towers from a bulk OpenCelliD CSV download and filters to the bounding box.
    Handles both plain CSV and .csv.gz compressed files.

    Args:
        csv_path: Path to the bulk CSV or .csv.gz file
        bbox: Geographic bounding box
        radio_filter: Optional list of radio types to include
        mcc_filter: Optional list of MCC values to include

    Returns:
        Path to the filtered output CSV.
    """
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, "towers_filtered.csv")

    logger.info(f"Loading bulk tower CSV: {csv_path}")

    opener = gzip.open if csv_path.endswith(".gz") else open

    rows = []
    with opener(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
            except (ValueError, TypeError):
                continue

            # Bounding box filter
            if not (bbox["min_lat"] <= lat <= bbox["max_lat"] and
                    bbox["min_lon"] <= lon <= bbox["max_lon"]):
                continue

            # Radio filter
            radio = row.get("radio", "")
            if radio_filter and radio not in radio_filter:
                continue

            # MCC filter
            mcc = row.get("mcc", "")
            if mcc_filter and mcc not in [str(m) for m in mcc_filter]:
                continue

            rows.append(row)

    if not rows:
        logger.warning("No towers found in bulk CSV matching the bounding box and filters.")
    else:
        logger.info(f"Extracted {len(rows)} towers from bulk CSV")

    # Write filtered output
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write("radio,mcc,net,area,cell,unit,lon,lat,range,samples,changeable,created,updated,averageSignal\n")

    return output_path


def load_towers(towers_path: str) -> list[dict]:
    """
    Loads tower data from CSV into a list of dicts.
    Each dict has: radio, mcc, net, lat, lon, range, averageSignal
    """
    towers = []
    if not os.path.exists(towers_path):
        logger.warning(f"Tower file not found: {towers_path}")
        return towers

    with open(towers_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tower = {
                    "radio": row.get("radio", "LTE"),
                    "mcc": row.get("mcc", ""),
                    "mnn": row.get("net", ""),
                    "lat": float(row.get("lat", 0)),
                    "lon": float(row.get("lon", 0)),
                    "range_m": float(row.get("range", 1000)) if row.get("range") else 1000.0,
                    "samples": int(row.get("samples", 1)) if row.get("samples") else 1,
                    "avg_signal": float(row.get("averageSignal", 0)) if row.get("averageSignal") else None,
                }
                towers.append(tower)
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed tower row: {e}")

    logger.info(f"Loaded {len(towers)} towers from {towers_path}")
    return towers


def _count_csv_rows(path: str) -> int:
    """Counts data rows in a CSV (excluding header)."""
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _filter_towers_by_mcc(path: str, mcc_filter: list):
    """In-place filter the tower CSV to only keep rows matching mcc_filter."""
    mcc_strs = [str(m) for m in mcc_filter]
    rows_in = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("mcc", "") in mcc_strs:
                rows_in.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_in)

    logger.info(f"Filtered towers to MCC {mcc_filter}: {len(rows_in)} towers remain")
