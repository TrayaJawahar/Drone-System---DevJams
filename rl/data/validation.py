import os
import sys
import yaml
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

    logger.info("Validation successful! Grid and columns are correct.")
    summary = get_data_quality_summary(df)
    print_quality_summary_report(summary)
    sys.exit(0)

if __name__ == "__main__":
    main()
