import os
import sys
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.utils.logger import setup_logger

logger = setup_logger("route_inference")

class RouteInferenceEngine:
    """
    Runs deterministic route planning inferences using a trained PPO model.
    Implements environment normalization alignment and infinite loop detection.
    """
    def __init__(
        self,
        config_path: str = "rl/config/rl_config.yaml",
        model_path: str = "rl/models/best_model/best_model.zip",
        vecnormalize_path: str = "rl/models/best_model/best_model_vecnormalize.pkl",
        feature_processor_path: str = "rl/models/best_model/feature_processor.joblib"
    ):
        # 1. Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 2. Check Data
        map_path = self.config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
        metadata_path = self.config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")
        self.df, self.metadata = load_geo_network_map(map_path, metadata_path)

        # 3. Load Feature Processor
        if not os.path.exists(feature_processor_path):
            raise FileNotFoundError(f"Feature processor not found at {feature_processor_path}")
        self.feature_processor = FeatureProcessor()
        self.feature_processor.load(feature_processor_path)

        # 4. Create Environment
        self.raw_env = DroneNetworkEnv(
            grid_df=self.df,
            metadata=self.metadata,
            config=self.config,
            feature_processor=self.feature_processor
        )
        self.mon_env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.mon_env])

        # 5. Load VecNormalize stats
        if not os.path.exists(vecnormalize_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vecnormalize_path}")
        self.norm_env = VecNormalize.load(vecnormalize_path, self.vec_env)
        self.norm_env.training = False
        self.norm_env.norm_reward = False

        # 6. Load PPO Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO Model not found at {model_path}")
        self.model = PPO.load(model_path, env=self.norm_env)
        
        # Loop visit limit
        self.loop_threshold = 5

    def generate_route(self, start: tuple[int, int], goal: tuple[int, int]) -> dict:
        """
        Generates a deterministic network-aware PPO path from start to goal.
        Tracks cell visits and breaks if infinite loops are detected.
        """
        logger.info(f"Generating PPO route from {start} to {goal}...")
        
        # Reset environment with custom start and goal coordinates using env_method
        try:
            self.norm_env.env_method("set_eval_mode", [(start, goal)])
            obs = self.norm_env.reset()
        except Exception as e:
            return {
                "success": False,
                "route_cells": [],
                "total_steps": 0,
                "total_distance": 0.0,
                "average_network_quality": 0.0,
                "outage_steps": 0,
                "battery_used": 100.0,
                "failure_reason": f"Reset failed: {e}"
            }

        route_cells = [start]
        visit_counts = {start: 1}
        
        success = False
        failure_reason = None
        info = {}

        max_steps = self.raw_env.max_steps
        for step in range(max_steps):
            # Predict deterministic action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = self.norm_env.step(action)
            info = infos[0]  # DummyVecEnv index 0
            
            curr_pos = info["current_pos"]
            route_cells.append(curr_pos)
            
            # Track loop visits
            visit_counts[curr_pos] = visit_counts.get(curr_pos, 0) + 1
            if visit_counts[curr_pos] > self.loop_threshold:
                failure_reason = "loop_detected"
                logger.warning(f"Inference loop detected at {curr_pos}. Halting generation.")
                break

            if dones[0]:
                success = info.get("success", False)
                failure_reason = info.get("failure_reason")
                break

        # If loop detected, clean termination status
        if failure_reason == "loop_detected":
            success = False

        # Gather final metrics
        total_steps = len(route_cells) - 1
        
        # Fetch detailed metrics from info if success, or calculate
        avg_net = info.get("avg_network_score", 0.0)
        outage_steps = info.get("total_outage_steps", 0)
        battery_used = float(self.raw_env.initial_battery - info.get("battery", 0.0))
        route_dist = info.get("route_distance", 0.0)

        # Fallback metric calculations if loop halted early before terminal step
        if failure_reason == "loop_detected":
            # Recompute route metrics manually
            from rl.evaluation.comparison import calculate_route_metrics
            calc = calculate_route_metrics(route_cells, self.raw_env.cell_lookup, self.raw_env.net_calc)
            avg_net = calc["avg_network_quality"]
            outage_steps = calc["outage_steps"]
            battery_used = calc["battery_used"]
            route_dist = calc["distance"]

        return {
            "success": bool(success),
            "route_cells": route_cells,
            "total_steps": int(total_steps),
            "total_distance": float(route_dist),
            "average_network_quality": float(avg_net),
            "outage_steps": int(outage_steps),
            "battery_used": float(battery_used),
            "failure_reason": failure_reason
        }

if __name__ == "__main__":
    # Check data policy and files
    config_path = "rl/config/rl_config.yaml"
    model_path = "rl/models/best_model/best_model.zip"
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    map_path = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")

    if not os.path.exists(map_path) or not os.path.exists(metadata_path):
        print("\nERROR: Real Geo-Network Map not found.")
        print(f"Please build {map_path} before training.\n")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"\nERROR: Trained PPO model not found at {model_path}.")
        print("Please train the model first by running: python run_training.py\n")
        sys.exit(1)

    print("Initializing RouteInferenceEngine...")
    try:
        engine = RouteInferenceEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        sys.exit(1)

    # Pick a mission from test scenarios if available, else generate a random one
    test_json = "rl/evaluation/test_scenarios.json"
    if os.path.exists(test_json):
        with open(test_json, "r") as f:
            scenarios = json.load(f)
        first_sc = scenarios[0]
        start = (first_sc["start"][0], first_sc["start"][1])
        goal = (first_sc["goal"][0], first_sc["goal"][1])
        print(f"Loaded mission from {test_json}: {start} -> {goal}")
    else:
        # Generate random valid mission
        start, goal = engine.raw_env.mission_gen.generate_random_mission(min_distance=20.0, max_distance=100.0)
        print(f"Generated random mission: {start} -> {goal}")

    print("Running PPO route inference...")
    res = engine.generate_route(start, goal)
    
    print("\n" + "=" * 40)
    print("ROUTE INFERENCE RESULTS")
    print("=" * 40)
    print(f"Success:             {res['success']}")
    print(f"Total Steps:         {res['total_steps']}")
    print(f"Total Distance:      {res['total_distance']:.2f}")
    print(f"Avg Network Quality: {res['average_network_quality']:.3f}")
    print(f"Outage Steps:        {res['outage_steps']}")
    print(f"Battery Used:        {res['battery_used']:.1f}%")
    if res['failure_reason']:
        print(f"Failure Reason:      {res['failure_reason']}")
    print(f"Route Cells:         {res['route_cells']}")
    print("=" * 40 + "\n")

