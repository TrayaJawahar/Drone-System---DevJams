import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rl.utils.logger import setup_logger

logger = setup_logger("feature_processor")

class FeatureProcessor:
    """
    Fits and applies scaling to observations for the PPO agent.
    Maintains availability masks for network metrics and handles missing values.
    """
    def __init__(self):
        self.scalers = {}
        # Define columns and corresponding scaler types
        self.features_config = {
            "grid_x": "minmax",
            "grid_y": "minmax",
            "latitude": "standard",
            "longitude": "standard",
            "elevation": "standard",
            "slope": "standard",
            "obstacle_distance": "minmax",
            "nearest_tower_distance": "minmax",
            "rssi": "standard",
            "rsrp": "standard",
            "sinr": "standard",
            "latency": "standard",
            "packet_loss": "standard",
            "throughput": "standard"
        }
        # Default raw values when metric is unavailable
        self.default_raw_values = {
            "rssi": -110.0,
            "rsrp": -140.0,
            "sinr": -10.0,
            "latency": 500.0,      # High latency default
            "packet_loss": 1.0,     # 100% loss default
            "throughput": 0.0,
            "elevation": 0.0,
            "slope": 0.0,
            "obstacle_distance": 0.0,
            "nearest_tower_distance": 1000.0
        }

    def fit(self, df: pd.DataFrame, metadata: dict = None):
        """
        Fits scalers on the available non-null data of the Geo-Network Map.
        """
        logger.info("Fitting feature scalers...")
        
        # Fit coordinates based on metadata dimensions if available, else max values
        if metadata and "grid_width" in metadata and "grid_height" in metadata:
            width = metadata["grid_width"]
            height = metadata["grid_height"]
            
            x_scaler = MinMaxScaler()
            x_scaler.fit(np.array([[0], [width - 1]]))
            self.scalers["grid_x"] = x_scaler
            
            y_scaler = MinMaxScaler()
            y_scaler.fit(np.array([[0], [height - 1]]))
            self.scalers["grid_y"] = y_scaler
            
            logger.info(f"Fitted grid_x and grid_y scalers using metadata dimensions: {width}x{height}")
        else:
            # Fallback to df min/max
            for col in ["grid_x", "grid_y"]:
                scaler = MinMaxScaler()
                scaler.fit(df[[col]].values)
                self.scalers[col] = scaler

        # Fit all other features on non-null data
        for col, scaler_type in self.features_config.items():
            if col in ["grid_x", "grid_y"]:
                continue  # Already processed
                
            if col in df.columns:
                non_null_data = df[col].dropna().values.reshape(-1, 1)
                if len(non_null_data) > 0:
                    if scaler_type == "minmax":
                        scaler = MinMaxScaler()
                    else:
                        scaler = StandardScaler()
                    scaler.fit(non_null_data)
                    self.scalers[col] = scaler
                    logger.debug(f"Fitted scaler for feature: {col}")
                else:
                    logger.warning(f"No non-null data found for '{col}'. Using dummy standard scaler.")
                    scaler = StandardScaler()
                    scaler.mean_ = np.array([0.0])
                    scaler.scale_ = np.array([1.0])
                    self.scalers[col] = scaler
            else:
                logger.warning(f"Column '{col}' not found in DataFrame. Using dummy standard scaler.")
                scaler = StandardScaler()
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
                self.scalers[col] = scaler

    def transform_value(self, feature_name: str, value) -> float:
        """
        Transforms a single feature value using its fitted scaler.
        Handles null or nan values by returning a normalized default (0.0).
        """
        if value is None or pd.isnull(value):
            return 0.0
            
        scaler = self.scalers.get(feature_name)
        if scaler is None:
            return float(value)
            
        try:
            val_arr = np.array([[float(value)]])
            scaled_val = scaler.transform(val_arr)[0, 0]
            return float(scaled_val)
        except Exception:
            return 0.0

    def get_feature_and_mask(self, feature_name: str, value) -> tuple[float, float]:
        """
        Returns (normalized_value, availability_mask) for a feature.
        If value is null or missing, uses standard default normalized to its scaler.
        """
        if value is None or pd.isnull(value):
            # Transform the default raw value
            default_raw = self.default_raw_values.get(feature_name, 0.0)
            norm_val = self.transform_value(feature_name, default_raw)
            return norm_val, 0.0
        else:
            norm_val = self.transform_value(feature_name, value)
            return norm_val, 1.0

    def save(self, file_path: str):
        """
        Saves the fitted scalers.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.scalers, file_path)
        logger.info(f"FeatureProcessor scalers saved to {file_path}")

    def load(self, file_path: str):
        """
        Loads the saved scalers.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No FeatureProcessor scalers found at {file_path}")
        self.scalers = joblib.load(file_path)
        logger.info(f"FeatureProcessor scalers loaded from {file_path}")
