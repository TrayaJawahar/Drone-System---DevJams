"""
network_estimator.py
--------------------
Estimates signal strength and network quality for each grid cell
using real cellular tower locations from OpenCelliD.

Model: ITU-R log-distance path-loss model
  PathLoss(dB) = 20*log10(4π*d*f/c) + n*(10*log10(d/d0)) + shadow_margin
  where:
    d = distance from tower (meters)
    f = carrier frequency (Hz)
    n = path-loss exponent (2.0 free space, 3.5 urban)
    shadow_margin = additional attenuation (dB)

Signal quality metrics computed:
  - nearest_tower_distance (meters) — real, from GPS coordinates
  - rssi (dBm) — estimated from path-loss model
  - rsrp (dBm) — LTE Reference Signal Received Power approximation
  - sinr (dB) — Signal-to-Interference + Noise Ratio
  - throughput (Mbps) — Shannon capacity estimate
  - network_data_confidence (0-1) — based on tower density + radio type
  - network_source = "tower_estimated"

Note: latency and packet_loss remain NaN.
These require real drive-test measurements or a network simulator.
"""

import math
import numpy as np
import pandas as pd
from rl.utils.logger import setup_logger

logger = setup_logger("network_estimator")

# Speed of light (m/s)
C = 3e8

# Boltzmann constant (W/Hz/K)
BOLTZMANN = 1.38e-23

# Receiver temperature (Kelvin)
TEMP_K = 290.0

# Radio type quality weights (higher = better network quality)
RADIO_QUALITY_WEIGHTS = {
    "NR": 1.0,
    "LTE": 0.85,
    "UMTS": 0.55,
    "GSM": 0.30,
}


def compute_path_loss_db(distance_m: float, freq_mhz: float,
                          path_loss_exp: float, shadow_margin_db: float) -> float:
    """
    Computes free-space + urban path loss in dB using the log-distance model.

    PL(d) = 20*log10(4π*d0*f/c) + 10*n*log10(d/d0) + shadow_margin
    where d0 = 1m reference distance.
    """
    if distance_m <= 0:
        return 0.0

    freq_hz = freq_mhz * 1e6
    d0 = 1.0  # reference distance (1 meter)

    # Free-space path loss at d0
    pl_d0 = 20 * math.log10(4 * math.pi * d0 * freq_hz / C)

    # Additional log-distance attenuation
    if distance_m > d0:
        pl_extra = 10 * path_loss_exp * math.log10(distance_m / d0)
    else:
        pl_extra = 0.0

    return pl_d0 + pl_extra + shadow_margin_db


def estimate_rssi(tx_power_dbm: float, path_loss_db: float,
                  min_rssi_dbm: float = -120.0) -> float:
    """Estimates RSSI = Tx Power - Path Loss (dBm)."""
    rssi = tx_power_dbm - path_loss_db
    return max(rssi, min_rssi_dbm)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes distance in meters between two GPS coordinates using Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_network_for_grid(grid_df: pd.DataFrame, towers: list,
                               config: dict) -> pd.DataFrame:
    """
    Estimates network quality metrics for each grid cell based on nearby towers.

    Args:
        grid_df: DataFrame with latitude, longitude columns
        towers: list of tower dicts (from tower_collector.load_towers)
        config: pipeline config dict

    Returns:
        Updated grid_df with added columns:
            nearest_tower_distance, rssi, rsrp, sinr, throughput,
            network_data_confidence, network_source, tower_count_nearby
    """
    net_cfg = config.get("network_estimation", {})
    path_loss_exp = net_cfg.get("path_loss_exponent", 3.5)
    noise_floor_dbm = net_cfg.get("noise_floor_dbm", -100.0)
    min_rssi_dbm = net_cfg.get("min_rssi_dbm", -120.0)
    search_radius_m = config.get("towers", {}).get("search_radius_meters", 5000)

    tx_power_cfg = net_cfg.get("tx_power", {})
    freq_cfg = net_cfg.get("frequency_mhz", {})
    shadow_cfg = net_cfg.get("shadow_margin_db", {})
    bw_cfg = net_cfg.get("bandwidth_mhz", {})

    n = len(grid_df)

    # Output arrays
    nearest_dist_arr = np.full(n, np.nan)
    rssi_arr = np.full(n, np.nan)
    rsrp_arr = np.full(n, np.nan)
    sinr_arr = np.full(n, np.nan)
    throughput_arr = np.full(n, np.nan)
    confidence_arr = np.full(n, 0.0)
    tower_count_arr = np.zeros(n, dtype=int)
    source_arr = ["missing"] * n

    if not towers:
        logger.warning("No towers available. All network metrics will be NaN.")
        grid_df = grid_df.copy()
        grid_df["nearest_tower_distance"] = np.nan
        grid_df["rssi"] = np.nan
        grid_df["rsrp"] = np.nan
        grid_df["sinr"] = np.nan
        grid_df["throughput"] = np.nan
        grid_df["latency"] = np.nan
        grid_df["packet_loss"] = np.nan
        grid_df["network_data_confidence"] = 0.0
        grid_df["tower_count_nearby"] = 0
        grid_df["network_source"] = "missing"
        return grid_df

    logger.info(f"Estimating network quality for {n} cells using {len(towers)} towers...")

    for i, row in grid_df.iterrows():
        cell_lat = row["latitude"]
        cell_lon = row["longitude"]

        # Find all towers within search radius
        nearby_towers = []
        for tower in towers:
            dist = haversine_distance(cell_lat, cell_lon, tower["lat"], tower["lon"])
            if dist <= search_radius_m:
                nearby_towers.append((dist, tower))

        if not nearby_towers:
            # No towers within radius → network metrics remain NaN
            source_arr[i] = "missing"
            continue

        # Sort by distance
        nearby_towers.sort(key=lambda x: x[0])
        tower_count_arr[i] = len(nearby_towers)
        nearest_dist_m, nearest_tower = nearby_towers[0]
        nearest_dist_arr[i] = nearest_dist_m

        radio = nearest_tower.get("radio", "LTE").upper()
        if radio not in RADIO_QUALITY_WEIGHTS:
            radio = "LTE"  # Default fallback

        tx_power = tx_power_cfg.get(radio, 43.0)
        freq_mhz = freq_cfg.get(radio, 2100.0)
        shadow_margin = shadow_cfg.get(radio, 8.0)
        bandwidth = bw_cfg.get(radio, 20.0)

        # --- RSSI estimation ---
        pl = compute_path_loss_db(nearest_dist_m, freq_mhz, path_loss_exp, shadow_margin)
        rssi = estimate_rssi(tx_power, pl, min_rssi_dbm)
        rssi_arr[i] = round(rssi, 2)

        # --- RSRP (LTE-specific; approximation for others) ---
        # RSRP = RSSI - 10*log10(12 * subcarrier_bandwidth_factor)
        # For simplicity: RSRP ≈ RSSI - 9dB (typical reference signal overhead)
        rsrp = rssi - 9.0 if radio in ("LTE", "NR") else rssi
        rsrp_arr[i] = round(rsrp, 2)

        # --- SINR (Signal to Interference + Noise Ratio) ---
        # Signal power (linear)
        signal_power_mw = 10 ** (rssi / 10)

        # Interference from other towers (sum of their RSSI contributions)
        interference_mw = 0.0
        for dist, interferer in nearby_towers[1:]:  # skip strongest (serving)
            i_radio = interferer.get("radio", "LTE").upper()
            if i_radio not in RADIO_QUALITY_WEIGHTS:
                i_radio = "LTE"
            i_tx = tx_power_cfg.get(i_radio, 43.0)
            i_freq = freq_cfg.get(i_radio, 2100.0)
            i_shadow = shadow_cfg.get(i_radio, 8.0)
            i_pl = compute_path_loss_db(dist, i_freq, path_loss_exp, i_shadow)
            i_rssi = estimate_rssi(i_tx, i_pl, min_rssi_dbm)
            interference_mw += 10 ** (i_rssi / 10)

        # Thermal noise power (dBm)
        freq_hz = freq_mhz * 1e6
        bw_hz = bandwidth * 1e6
        noise_mw = 10 ** (noise_floor_dbm / 10)

        sinr_linear = signal_power_mw / (interference_mw + noise_mw + 1e-12)
        sinr_db = 10 * math.log10(sinr_linear + 1e-12)
        sinr_arr[i] = round(sinr_db, 2)

        # --- Throughput (Shannon capacity) ---
        # C = B * log2(1 + SNR)
        snr_linear = signal_power_mw / (noise_mw + 1e-12)
        throughput_bps = bandwidth * 1e6 * math.log2(1 + snr_linear)
        throughput_mbps = throughput_bps / 1e6
        # Apply radio-type efficiency factor (overhead, modulation limits)
        efficiency = {
            "NR": 0.75, "LTE": 0.65, "UMTS": 0.40, "GSM": 0.10
        }.get(radio, 0.65)
        throughput_arr[i] = round(min(throughput_mbps * efficiency, 1000.0), 2)  # cap at 1Gbps

        # --- Confidence score ---
        # Based on: tower density, radio type quality, signal strength
        radio_quality = RADIO_QUALITY_WEIGHTS.get(radio, 0.5)
        # Normalize tower count (more towers = higher confidence up to ~5)
        density_score = min(tower_count_arr[i] / 5.0, 1.0)
        # Normalize distance (closer = higher confidence)
        dist_score = max(0.0, 1.0 - (nearest_dist_m / search_radius_m))
        confidence = 0.4 * radio_quality + 0.3 * density_score + 0.3 * dist_score
        confidence_arr[i] = round(confidence, 3)

        source_arr[i] = "tower_estimated"

        if (i + 1) % 500 == 0:
            logger.info(f"  Estimated {i + 1}/{n} cells...")

    grid_df = grid_df.copy()
    grid_df["nearest_tower_distance"] = np.where(np.isnan(nearest_dist_arr), np.nan, nearest_dist_arr)
    grid_df["rssi"] = rssi_arr
    grid_df["rsrp"] = rsrp_arr
    grid_df["sinr"] = sinr_arr
    grid_df["throughput"] = throughput_arr
    grid_df["latency"] = np.nan          # Requires real measurements
    grid_df["packet_loss"] = np.nan      # Requires real measurements
    grid_df["network_data_confidence"] = confidence_arr
    grid_df["tower_count_nearby"] = tower_count_arr
    grid_df["network_source"] = source_arr

    estimated_count = sum(1 for s in source_arr if s == "tower_estimated")
    logger.info(f"Network estimation complete: {estimated_count}/{n} cells have coverage ({100*estimated_count/n:.1f}%)")

    return grid_df
