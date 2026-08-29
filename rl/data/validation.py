import os
import sys
import yaml
import numpy as np
from rl.data.data_loader import load_geo_network_map, get_data_quality_summary, print_quality_summary_report
from rl.utils.logger import setup_logger

logger = setup_logger("validation")

def main():
    config_path = "rl/config/rl_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse configuration file {config_path}: {e}")
        sys.exit(1)

    map_path = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")

    logger.info(f"Validating Geo-Network Map at: {map_path}")
    logger.info(f"Using Metadata JSON at: {metadata_path}")

    try:
        df, metadata = load_geo_network_map(map_path, metadata_path)
    except FileNotFoundError as e:
        print(f"\n{e}\n")
        logger.error(f"Validation failed: Files not found.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation failed: Data format error. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: Unexpected error: {e}")
        sys.exit(1)

    # --- STRICT VALIDATION GATE ---
    critical_columns = ["elevation", "slope", "nearest_tower_distance", "rssi", "rsrp", "sinr"]
    failed_columns = []

    for col in critical_columns:
        if col not in df.columns:
            failed_columns.append(f"{col}: column completely missing")
        else:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > 0.99:
                failed_columns.append(f"{col}: {nan_ratio*100:.1f}% missing")

    # Check network confidence variance
    if "network_data_confidence" in df.columns:
        if df["network_data_confidence"].std() < 0.001:
            failed_columns.append("network_data_confidence: zero variance (flat score)")

    if failed_columns:
        print("\n" + "="*60)
        print("DATA VALIDATION FAILED")
        print("="*60)
        for fail in failed_columns:
            print(f"  {fail}")
        print("\nPPO training aborted because the environment dataset is invalid.")
        print("="*60 + "\n")
        sys.exit(1)

    logger.info("Validation successful! Grid, columns, and terrain/network data are valid.")
    summary = get_data_quality_summary(df)
    print_quality_summary_report(summary)
    sys.exit(0)

if __name__ == "__main__":
    main()
