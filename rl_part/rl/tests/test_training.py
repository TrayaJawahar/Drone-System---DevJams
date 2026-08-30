import os
import tempfile
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.tests.fixtures.mock_data import mock_dataset_paths
from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.inference.route_inference import RouteInferenceEngine

TEST_TRAIN_CONFIG = {
    "project": {"name": "test"},
    "data": {},
    "environment": {
        "max_steps": 10,
        "battery": {
            "initial": 100.0,
            "horizontal_cost": 0.5,
            "diagonal_cost": 0.7,
            "slope_factor": 0.05
        }
    },
    "observation": {
        "include_neighbor_network": True,
        "include_diagonal_neighbors": False,
        "include_terrain": True
    },
    "reward": {
        "progress_weight": 5.0,
        "network_weight": 3.0,
        "safety_weight": 2.0,
        "movement_penalty": 0.1,
        "energy_penalty": 0.05,
        "outage_penalty": 5.0,
        "collision_penalty": 100.0,
        "goal_reward": 100.0,
        "timeout_penalty": 10.0
    },
    "network": {
        "outage_threshold": 0.3,
        "min_confidence": 0.5
    },
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 64,
        "batch_size": 16,
        "n_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }
}

def test_environment_compatibility(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, TEST_TRAIN_CONFIG, fp)
    # Check compliance with Gym API
    check_env(env)

def test_short_ppo_train_and_inference(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    # Create save directory
    temp_dir = tempfile.mkdtemp()
    fp_path = os.path.join(temp_dir, "feature_processor.joblib")
    fp.save(fp_path)
    
    # 1. Setup Train environment
    raw_env = DroneNetworkEnv(df, metadata, TEST_TRAIN_CONFIG, fp)
    mon_env = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: mon_env])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    
    # 2. Setup PPO
    model = PPO(
        policy="MlpPolicy",
        env=norm_env,
        n_steps=64,
        batch_size=16,
        n_epochs=2,
        verbose=0
    )
    
    # 3. Train a tiny bit
    model.learn(total_timesteps=64)
    
    # 4. Save model and stats
    model_path = os.path.join(temp_dir, "best_model.zip")
    vec_path = os.path.join(temp_dir, "best_model_vecnormalize.pkl")
    
    model.save(model_path)
    norm_env.save(vec_path)
    
    assert os.path.exists(model_path)
    assert os.path.exists(vec_path)
    
    # 5. Load model & verify inference
    # Create temp config YAML
    config_copy = dict(TEST_TRAIN_CONFIG)
    config_copy["data"] = {
        "geo_network_map": map_path,
        "metadata": metadata_path
    }
    
    config_path = os.path.join(temp_dir, "rl_config.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config_copy, f)
        
    engine = RouteInferenceEngine(
        config_path=config_path,
        model_path=model_path,
        vecnormalize_path=vec_path,
        feature_processor_path=fp_path
    )
    
    # Generate route on a valid scenario: start (1, 1) to goal (2, 2)
    res = engine.generate_route((1, 1), (2, 2))
    
    assert "success" in res
    assert "route_cells" in res
    assert len(res["route_cells"]) >= 2
    assert res["route_cells"][0] == (1, 1)

    # Clean up temp files
    try:
        os.remove(model_path)
        os.remove(vec_path)
        os.remove(fp_path)
        os.remove(config_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
