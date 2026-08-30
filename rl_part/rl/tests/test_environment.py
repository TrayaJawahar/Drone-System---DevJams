import pytest
import numpy as np
from rl.tests.fixtures.mock_data import mock_dataset_paths
from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.environment.actions import ACTION_MAP

# Basic testing config
ENV_CONFIG = {
    "project": {"name": "test"},
    "data": {},
    "environment": {
        "max_steps": 10,
        "battery": {
            "initial": 100.0,
            "horizontal_cost": 5.0, # High cost for easy depletion testing
            "diagonal_cost": 7.0,
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
    }
}

def test_environment_reset_and_dims(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, ENV_CONFIG, fp)
    
    # 1. Reset check
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "start" in info
    assert "goal" in info
    assert env.battery == 100.0
    assert env.step_count == 0

def test_valid_movement_and_battery(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, ENV_CONFIG, fp)
    env.reset(options={"start": (1, 1), "goal": (8, 8)})
    
    # Move East (dx=1, dy=0) -> action 2
    # From (1,1) to (2,1). Not an obstacle.
    obs, reward, terminated, truncated, info = env.step(2)
    
    assert env.current_pos == (2, 1)
    assert env.step_count == 1
    # Battery depletion check: 100.0 - 5.0 (base cost) - slope_cost
    # Slope for column 2 is 2 * 0.5 = 1.0. Slope cost = 1.0 * 0.05 = 0.05.
    # Total battery used = 5.05. Remaining = 100 - 5.05 = 94.95.
    assert pytest.approx(env.battery, 0.001) == 94.95
    assert not terminated
    assert not truncated

def test_boundary_violation(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, ENV_CONFIG, fp)
    # Start at boundary (0, 0)
    env.reset(options={"start": (0, 0), "goal": (8, 8)})
    
    # Move West (action 3: dx=-1, dy=0)
    # Out of grid boundary!
    obs, reward, terminated, truncated, info = env.step(3)
    
    # Drone must stay in the same cell and not crash/terminate, but still consume energy
    assert env.current_pos == (0, 0)
    assert not terminated
    assert env.battery == 95.0 # Consumed 5.0 battery

def test_obstacle_collision(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, ENV_CONFIG, fp)
    # Column 4 has obstacles except row 2 and 7.
    # Set start near obstacle: start (3, 3), goal (8, 8). Move East (action 2) to collide with (4, 3) obstacle.
    env.reset(options={"start": (3, 3), "goal": (8, 8)})
    
    obs, reward, terminated, truncated, info = env.step(2)
    
    assert terminated
    assert info["collision"]
    assert info["failure_reason"] == "collision"
    assert reward < -50.0  # Large collision penalty

def test_goal_reached(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    env = DroneNetworkEnv(df, metadata, ENV_CONFIG, fp)
    # Start (1, 1), Goal (2, 1). Move East (action 2) to reach goal.
    env.reset(options={"start": (1, 1), "goal": (2, 1)})
    
    obs, reward, terminated, truncated, info = env.step(2)
    
    assert terminated
    assert info["success"]
    assert info["failure_reason"] is None
    assert reward > 50.0  # Large goal reward

def test_battery_depletion(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    # Configure environment with very high step costs to deplete battery in 2 steps
    deplete_config = dict(ENV_CONFIG)
    deplete_config["environment"] = {
        "max_steps": 10,
        "battery": {
            "initial": 10.0,
            "horizontal_cost": 6.0,
            "diagonal_cost": 8.0,
            "slope_factor": 0.0
        }
    }
    
    env = DroneNetworkEnv(df, metadata, deplete_config, fp)
    env.reset(options={"start": (1, 1), "goal": (8, 8)})
    
    # Step 1: battery becomes 10.0 - 6.0 = 4.0
    obs, reward, terminated, truncated, info = env.step(2)
    assert not terminated
    
    # Step 2: battery becomes 4.0 - 6.0 = -2.0 -> 0.0 (Depleted!)
    obs, reward, terminated, truncated, info = env.step(2)
    assert terminated
    assert info["battery_depleted"]
    assert env.battery == 0.0
    assert info["failure_reason"] == "battery_depleted"

def test_maximum_steps(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    # Configure environment to time out in 2 steps
    timeout_config = dict(ENV_CONFIG)
    timeout_config["environment"] = {
        "max_steps": 2,
        "battery": {
            "initial": 100.0,
            "horizontal_cost": 0.1,
            "diagonal_cost": 0.1,
            "slope_factor": 0.0
        }
    }
    
    env = DroneNetworkEnv(df, metadata, timeout_config, fp)
    env.reset(options={"start": (1, 1), "goal": (8, 8)})
    
    # Step 1
    obs, reward, terminated, truncated, info = env.step(2)
    assert not terminated and not truncated
    
    # Step 2
    obs, reward, terminated, truncated, info = env.step(2)
    assert truncated
    assert info["timeout"]
    assert info["failure_reason"] == "timeout"
