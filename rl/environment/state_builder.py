import numpy as np
import pandas as pd
from rl.environment.actions import ACTION_MAP
from rl.utils.logger import setup_logger

logger = setup_logger("state_builder")

class StateBuilder:
    """
    Constructs a normalized, fixed-length state observation vector for the RL agent.
    Supports neighboring network metrics and terrain features depending on configuration.
    """
    def __init__(self, config: dict):
        self.config = config
        self.obs_config = config.get("observation", {})
        
        self.include_neighbor_network = self.obs_config.get("include_neighbor_network", True)
        self.include_diagonal_neighbors = self.obs_config.get("include_diagonal_neighbors", False)
        self.include_terrain = self.obs_config.get("include_terrain", True)
        
        self.max_obstacle_distance = 10.0  # Max grid steps for radar lookahead
        self.observation_size = self._calculate_observation_size()
        logger.info(f"StateBuilder initialized. Observation vector size: {self.observation_size}")

    def _calculate_observation_size(self) -> int:
        """
        Determines the total size of the flat state vector.
        """
        # Core State:
        # 1. Current X (1)
        # 2. Current Y (1)
        # 3. Goal X (1)
        # 4. Goal Y (1)
        # 5. Delta X (1)
        # 6. Delta Y (1)
        # 7. Distance to goal (1)
        # 8. 8-direction obstacle information (8)
        # 9. Nearest obstacle distance (1)
        # 10. Current Network Quality (1)
        # 11. Network Data Confidence (1)
        # 12. Battery Remaining (1)
        size = 19

        # Optional Network Features (Value + Mask for 6 metrics)
        # rssi, rsrp, sinr, latency, packet_loss, throughput (6 * 2 = 12)
        size += 12

        # Terrain (Value + Mask for elevation and slope, 2 * 2 = 4)
        if self.include_terrain:
            size += 4

        # Tower distance (Value + Mask, 1 * 2 = 2)
        size += 2

        # Neighbor network quality and confidence
        if self.include_neighbor_network:
            num_neighbors = 8 if self.include_diagonal_neighbors else 4
            size += num_neighbors * 2  # (quality + confidence) per neighbor

        return size

    def get_observation_shape(self) -> tuple[int]:
        return (self.observation_size,)

    def get_obstacle_distances(self, current_pos: tuple[int, int], grid_free_mask: np.ndarray) -> np.ndarray:
        """
        Calculates normalized obstacle distances in 8 directions (North, South, East, West, and diagonals).
        Returns an array of 8 values in range [0.0, 1.0], where 0.0 means immediate obstacle.
        """
        cx, cy = current_pos
        w, h = grid_free_mask.shape
        obstacle_dists = np.zeros(8, dtype=np.float32)

        # Actions 0 to 7 correspond to the 8 directions in ACTION_MAP
        for action_idx in range(8):
            dx, dy = ACTION_MAP[action_idx]
            dist = self.max_obstacle_distance
            
            # Step in the direction until we hit an obstacle, bound, or reach max distance
            for step in range(1, int(self.max_obstacle_distance) + 1):
                nx, ny = cx + dx * step, cy + dy * step
                
                # Check grid boundaries
                if not (0 <= nx < w and 0 <= ny < h):
                    dist = float(step - 1)
                    break
                # Check obstacle
                if not grid_free_mask[nx, ny]:
                    dist = float(step - 1)
                    break
                    
            # Normalize to [0.0, 1.0]
            obstacle_dists[action_idx] = dist / self.max_obstacle_distance

        return obstacle_dists

    def build_state(
        self,
        current_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        battery: float,
        cell_lookup: dict,
        grid_free_mask: np.ndarray,
        feature_processor,
        network_quality_calculator
    ) -> np.ndarray:
        """
        Assembles and returns the flat normalized observation vector.
        """
        cx, cy = current_pos
        gx, gy = goal_pos
        w, h = grid_free_mask.shape

        # Fetch current cell data
        cell_data = cell_lookup.get((cx, cy), {})
        
        # 1. Normalized positions
        norm_cx = feature_processor.transform_value("grid_x", cx)
        norm_cy = feature_processor.transform_value("grid_y", cy)
        norm_gx = feature_processor.transform_value("grid_x", gx)
        norm_gy = feature_processor.transform_value("grid_y", gy)
        
        # 2. Goal Deltas & Distances
        dx = gx - cx
        dy = gy - cy
        # Scale deltas based on grid dimensions to keep them normalized
        norm_dx = dx / w
        norm_dy = dy / h
        
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        norm_distance = distance_to_goal / np.sqrt(w**2 + h**2)

        # 3. Obstacles radar
        obstacle_dists = self.get_obstacle_distances(current_pos, grid_free_mask)
        
        # Nearest obstacle distance overall (either from cell attribute or minimum radar distance)
        raw_obs_dist = cell_data.get("obstacle_distance")
        if raw_obs_dist is not None and not pd.isnull(raw_obs_dist):
            norm_nearest_obs = feature_processor.transform_value("obstacle_distance", raw_obs_dist)
        else:
            norm_nearest_obs = float(np.min(obstacle_dists))

        # 4. Network Quality Score & Confidence
        net_info = network_quality_calculator.calculate(cell_data)
        net_score = net_info["network_score"]
        net_confidence = net_info["confidence"]

        # 5. Battery (0.0 to 1.0)
        norm_battery = battery / 100.0

        # Assembly: Core State
        state = [
            norm_cx, norm_cy,
            norm_gx, norm_gy,
            norm_dx, norm_dy,
            norm_distance,
            *obstacle_dists,
            norm_nearest_obs,
            net_score,
            net_confidence,
            norm_battery
        ]

        # 6. Optional Network Features (Value + Mask)
        network_features = ["rssi", "rsrp", "sinr", "latency", "packet_loss", "throughput"]
        for feat in network_features:
            val, mask = feature_processor.get_feature_and_mask(feat, cell_data.get(feat))
            state.extend([val, mask])

        # 7. Terrain (Value + Mask)
        if self.include_terrain:
            elevation_val, elevation_mask = feature_processor.get_feature_and_mask("elevation", cell_data.get("elevation"))
            slope_val, slope_mask = feature_processor.get_feature_and_mask("slope", cell_data.get("slope"))
            state.extend([elevation_val, elevation_mask, slope_val, slope_mask])

        # 8. Tower Distance (Value + Mask)
        tower_dist_val, tower_dist_mask = feature_processor.get_feature_and_mask("nearest_tower_distance", cell_data.get("nearest_tower_distance"))
        state.extend([tower_dist_val, tower_dist_mask])

        # 9. Neighbor Network Quality
        if self.include_neighbor_network:
            # Directions to retrieve
            if self.include_diagonal_neighbors:
                directions = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                # Cardinal directions only
                directions = [0, 1, 2, 3] # North, South, East, West
                
            for act in directions:
                ndx, ndy = ACTION_MAP[act]
                neighbor_pos = (cx + ndx, cy + ndy)
                
                # Verify if neighbor is inside grid and not an obstacle
                if (0 <= neighbor_pos[0] < w and 0 <= neighbor_pos[1] < h) and grid_free_mask[neighbor_pos[0], neighbor_pos[1]]:
                    neighbor_cell_data = cell_lookup.get(neighbor_pos, {})
                    n_net_info = network_quality_calculator.calculate(neighbor_cell_data)
                    n_score = n_net_info["network_score"]
                    n_conf = n_net_info["confidence"]
                else:
                    n_score = 0.0
                    n_conf = 0.0
                    
                state.extend([n_score, n_conf])

        # Ensure correct observation size
        assert len(state) == self.observation_size, f"Observation size mismatch! Expected {self.observation_size}, got {len(state)}"

        return np.array(state, dtype=np.float32)
