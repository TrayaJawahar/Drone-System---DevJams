import os
import yaml
import numpy as np
import pandas as pd
import pytest
from rl.tests.fixtures.mock_data import mock_dataset_paths
from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.state_builder import StateBuilder
from rl.environment.network_quality import NetworkQualityCalculator
from rl.environment.mission_generator import MissionGenerator

# Load basic config for testing
TEST_CONFIG = {
    "observation": {
        "include_neighbor_network": True,
        "include_diagonal_neighbors": False,
        "include_terrain": True
    },
    "network": {
        "outage_threshold": 0.3,
        "min_confidence": 0.5
    }
}

def test_observation_shape_and_masks(mock_dataset_paths):
    map_path, metadata_path = mock_dataset_paths
    df, metadata = load_geo_network_map(map_path, metadata_path)
    
    # Fit FeatureProcessor
    fp = FeatureProcessor()
    fp.fit(df, metadata)
    
    net_calc = NetworkQualityCalculator()
    
    # 1. Test state builder observation size
    sb = StateBuilder(TEST_CONFIG)
    obs_shape = sb.get_observation_shape()
    
    # Core (19) + Metrics (12) + Terrain (4) + Tower (2) + Cardinal Neighbors (8) = 45 values
    expected_size = 19 + 12 + 4 + 2 + 8
    assert obs_shape == (expected_size,)

    # 2. Build state at cell (1, 1) - RSSI is None (unavailable)
    # Get cell lookup and grid mask
    mission_gen = MissionGenerator(df, metadata)
    cell_lookup = { (int(row["grid_x"]), int(row["grid_y"])): row.to_dict() for _, row in df.iterrows() }
    
    state = sb.build_state(
        current_pos=(1, 1),
        goal_pos=(8, 8),
        battery=100.0,
        cell_lookup=cell_lookup,
        grid_free_mask=mission_gen.grid_free_mask,
        feature_processor=fp,
        network_quality_calculator=net_calc
    )
    
    assert len(state) == expected_size
    
    # RSSI mask is index 19 (value) + 20 (mask), RSSI is at index 19 & 20
    # Let's count indices in state vector:
    # Core state has 19 elements, so index 0 to 18.
    # Optional network metrics start at index 19:
    # rssi: index 19, rssi_mask: index 20
    # Since column 1 has RSSI as None, rssi_mask (index 20) must be 0.0
    assert state[20] == 0.0
    
    # RSRP mask is index 21 + 22, RSRP is at index 21 & 22
    assert state[22] == 0.0
    
    # Cell (5, 5) has valid RSSI
    state_valid = sb.build_state(
        current_pos=(5, 5),
        goal_pos=(8, 8),
        battery=80.0,
        cell_lookup=cell_lookup,
        grid_free_mask=mission_gen.grid_free_mask,
        feature_processor=fp,
        network_quality_calculator=net_calc
    )
    # RSSI mask at index 20 must be 1.0 (available)
    assert state_valid[20] == 1.0

def test_neighbor_network_toggle():
    # Toggle include_neighbor_network = False
    config_no_neighbors = {
        "observation": {
            "include_neighbor_network": False,
            "include_diagonal_neighbors": False,
            "include_terrain": True
        }
    }
    
    sb = StateBuilder(config_no_neighbors)
    expected_size = 19 + 12 + 4 + 2 # No neighbor features (45 - 8 = 37)
    assert sb.get_observation_shape() == (expected_size,)
