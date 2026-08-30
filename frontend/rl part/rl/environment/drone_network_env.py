import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from rl.environment.actions import ACTION_MAP, is_diagonal
from rl.environment.mission_generator import MissionGenerator
from rl.environment.network_quality import NetworkQualityCalculator
from rl.environment.reward import RewardCalculator
from rl.environment.state_builder import StateBuilder
from rl.data.feature_processor import FeatureProcessor
from rl.utils.logger import setup_logger

logger = setup_logger("drone_network_env")

# All valid termination reason strings
TERM_GOAL_REACHED     = "goal_reached"
TERM_BATTERY_DEPLETED = "battery_depleted"
TERM_TIMEOUT          = "timeout"
TERM_COLLISION        = "collision"         # only if terminate_on_collision=True


class DroneNetworkEnv(gym.Env):
    """
    Gymnasium environment representing a drone navigating through a Geo-Network Map.
    Optimizes path navigation subject to obstacles, battery capacity, and signal outage constraints.

    Key design changes vs. original:
    - Collisions and boundary violations are NON-TERMINATING by default.
      The drone stays in its current cell, receives a penalty, and increments
      invalid_move_count.  This allows PPO to collect trajectories long enough
      to learn navigation.  Configurable via environment.terminate_on_collision.
    - Only terminates on: goal_reached | battery_depleted | max_steps.
    - Full termination reason logging in every episode's final info dict.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_df: pd.DataFrame,
        metadata: dict,
        config: dict,
        feature_processor: FeatureProcessor,
        network_quality_calculator: NetworkQualityCalculator = None,
        state_builder: StateBuilder = None,
        mission_generator: MissionGenerator = None
    ):
        super().__init__()
        self.grid_df = grid_df
        self.grid_metadata = metadata
        self.config = config
        self.feature_processor = feature_processor

        # Instantiate subcomponents if not provided
        self.net_calc = network_quality_calculator or NetworkQualityCalculator(config)
        self.state_builder = state_builder or StateBuilder(config)
        self.mission_gen = mission_generator or MissionGenerator(grid_df, metadata)
        self.reward_calc = RewardCalculator(config)

        # Environment Limits
        env_config = config.get("environment", {})
        self.max_steps = int(env_config.get("max_steps", 500))

        # ── Collision termination flag ──────────────────────────────────────
        # When False (default): collisions & boundary violations are invalid
        # moves — drone stays, gets penalty, episode continues.
        # When True: collision terminates the episode immediately.
        self.terminate_on_collision = bool(env_config.get("terminate_on_collision", False))

        # Battery Parameters
        bat_config = env_config.get("battery", {})
        self.initial_battery = float(bat_config.get("initial", 100.0))
        self.horizontal_cost = float(bat_config.get("horizontal_cost", 0.5))
        self.diagonal_cost   = float(bat_config.get("diagonal_cost", 0.7))
        self.slope_factor    = float(bat_config.get("slope_factor", 0.05))

        # Curriculum distance limits (updated dynamically during training)
        curr_stages = config.get("curriculum", {}).get("stages", [])
        if curr_stages:
            self.min_dist = float(curr_stages[0].get("min_distance", 10))
            self.max_dist = float(curr_stages[0].get("max_distance", 30))
        else:
            self.min_dist = 10.0
            self.max_dist = 200.0

        # Action Space: 8 discrete directions
        self.action_space = spaces.Discrete(8)

        # Observation Space: continuous vector
        obs_shape = self.state_builder.get_observation_shape()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # Fast lookup: (grid_x, grid_y) → row dict
        self.cell_lookup = {}
        for _, row in grid_df.iterrows():
            gx = int(row["grid_x"])
            gy = int(row["grid_y"])
            self.cell_lookup[(gx, gy)] = row.to_dict()

        # Grid bounds
        self.grid_width  = self.mission_gen.grid_width
        self.grid_height = self.mission_gen.grid_height

        # Episode state (initialised in reset())
        self.current_pos = (0, 0)
        self.goal_pos    = (0, 0)
        self.battery     = self.initial_battery
        self.step_count  = 0

        # Tracking
        self.consecutive_outage_steps = 0
        self.total_outage_steps       = 0
        self.invalid_move_count       = 0  # boundary + obstacle violations
        self.visited_cells            = set()
        self.current_route            = []
        self.steps_moving_closer      = 0
        self.steps_moving_away        = 0
        self.min_dist_reached         = float('inf')
        self.battery_cost_movement    = 0.0
        self.battery_cost_slope       = 0.0
        self.battery_cost_invalid     = 0.0

        # Eval mode
        self.eval_missions = None
        self.eval_index    = 0

        logger.info(
            f"DroneNetworkEnv initialised. "
            f"Grid: {self.grid_width}x{self.grid_height}  "
            f"terminate_on_collision={self.terminate_on_collision}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────────

    def set_curriculum_stage(self, min_dist: float, max_dist: float):
        self.min_dist = min_dist
        self.max_dist = max_dist
        logger.info(f"Curriculum updated: min_dist={min_dist}, max_dist={max_dist}")

    def set_eval_mode(self, missions: list):
        self.eval_missions = missions
        self.eval_index    = 0
        logger.info(f"Eval mode enabled with {len(missions)} fixed missions.")

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        fixed_mission = None
        if options and "start" in options and "goal" in options:
            fixed_mission = (options["start"], options["goal"])

        if fixed_mission:
            self.current_pos, self.goal_pos = self.mission_gen.generate_fixed_mission(
                fixed_mission[0], fixed_mission[1]
            )
        elif self.eval_missions is not None and len(self.eval_missions) > 0:
            start_goal = self.eval_missions[self.eval_index]
            self.current_pos, self.goal_pos = self.mission_gen.generate_fixed_mission(
                start_goal[0], start_goal[1]
            )
            self.eval_index = (self.eval_index + 1) % len(self.eval_missions)
        else:
            self.current_pos, self.goal_pos = self.mission_gen.generate_random_mission(
                self.min_dist, self.max_dist
            )

        # Reset episode state
        self.battery                  = self.initial_battery
        self.step_count               = 0
        self.consecutive_outage_steps = 0
        self.total_outage_steps       = 0
        self.invalid_move_count       = 0
        self.steps_moving_closer      = 0
        self.steps_moving_away        = 0
        self.battery_cost_movement    = 0.0
        self.battery_cost_slope       = 0.0
        self.battery_cost_invalid     = 0.0
        self.visited_cells            = {self.current_pos}
        self.current_route            = [self.current_pos]

        # Episode metric trackers
        self.episode_network_scores  = []
        self.episode_latencies       = []
        self.episode_packet_losses   = []
        self.episode_rssi_values     = []
        self._ep_start_pos           = self.current_pos
        self._ep_start_dist          = float(np.sqrt(
            (self.current_pos[0] - self.goal_pos[0])**2 +
            (self.current_pos[1] - self.goal_pos[1])**2
        ))
        self.min_dist_reached = self._ep_start_dist

        # Record initial cell metrics
        init_cell = self.cell_lookup.get(self.current_pos, {})
        init_net  = self.net_calc.calculate(init_cell)
        self.episode_network_scores.append(init_net["network_score"])
        self.episode_latencies.append(init_cell.get("latency"))
        self.episode_packet_losses.append(init_cell.get("packet_loss"))
        self.episode_rssi_values.append(init_cell.get("rssi"))

        obs = self._build_obs()
        info = {
            "start":      self.current_pos,
            "goal":       self.goal_pos,
            "battery":    self.battery,
            "step_count": self.step_count,
        }
        return obs, info

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: int):
        self.step_count += 1
        dx, dy = ACTION_MAP[action]
        cx, cy = self.current_pos
        nx, ny = cx + dx, cy + dy

        is_boundary_violation = False
        is_collision          = False

        # ── 1. Corner cutting check ──────────────────────────────────────────
        if dx != 0 and dy != 0:
            cx_valid = (0 <= cx + dx < self.grid_width and 0 <= cy < self.grid_height)
            cy_valid = (0 <= cx < self.grid_width and 0 <= cy + dy < self.grid_height)
            if (not cx_valid or not self.mission_gen.grid_free_mask[cx + dx, cy]) or \
               (not cy_valid or not self.mission_gen.grid_free_mask[cx, cy + dy]):
                is_collision = True
                nx, ny = cx, cy

        # ── 2. Boundary check ────────────────────────────────────────────────
        if not is_collision and not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
            is_boundary_violation = True
            nx, ny = cx, cy          # drone stays

        # ── 3. Obstacle check ────────────────────────────────────────────────
        elif not is_collision and not self.mission_gen.grid_free_mask[nx, ny]:
            is_collision = True
            nx, ny = cx, cy          # drone stays

        # ── 3. Invalid-move bookkeeping ──────────────────────────────────────
        is_invalid_move = is_boundary_violation or is_collision
        if is_invalid_move:
            self.invalid_move_count += 1

        # ── 4. Battery ───────────────────────────────────────────────────────
        base_cost  = self.diagonal_cost if is_diagonal(action) else self.horizontal_cost
        if is_invalid_move:
            step_energy_spent = base_cost * 0.5   # half cost for wasted move
            self.battery_cost_invalid += step_energy_spent
        else:
            cell_data = self.cell_lookup.get((nx, ny), {})
            slope     = cell_data.get("slope", 0.0)
            if pd.isnull(slope):
                slope = 0.0
            
            movement_cost = base_cost
            slope_cost = max(0.0, slope) * self.slope_factor
            step_energy_spent = movement_cost + slope_cost
            
            self.battery_cost_movement += movement_cost
            self.battery_cost_slope += slope_cost

        self.battery = max(0.0, self.battery - step_energy_spent)

        # ── 5. Move drone (only on valid moves) ──────────────────────────────
        if not is_invalid_move:
            self.current_pos = (nx, ny)
            self.current_route.append(self.current_pos)
            self.visited_cells.add(self.current_pos)

        # ── 6. Retrieve new cell data ─────────────────────────────────────────
        current_cell_data = self.cell_lookup.get(self.current_pos, {})
        net_info  = self.net_calc.calculate(current_cell_data)
        net_score = net_info["network_score"]

        self.episode_network_scores.append(net_score)
        self.episode_latencies.append(current_cell_data.get("latency"))
        self.episode_packet_losses.append(current_cell_data.get("packet_loss"))
        self.episode_rssi_values.append(current_cell_data.get("rssi"))

        # ── 7. Outage tracking ───────────────────────────────────────────────
        if net_score < self.net_calc.outage_threshold:
            self.consecutive_outage_steps += 1
            self.total_outage_steps       += 1
        else:
            self.consecutive_outage_steps = 0

        # ── 8. Distance metrics ──────────────────────────────────────────────
        gx, gy   = self.goal_pos
        prev_dist = float(np.sqrt((cx - gx)**2 + (cy - gy)**2))
        curr_dist = float(np.sqrt((self.current_pos[0] - gx)**2 + (self.current_pos[1] - gy)**2))
        
        self.min_dist_reached = min(self.min_dist_reached, curr_dist)
        
        if prev_dist - curr_dist > 0.001:
            self.steps_moving_closer += 1
        elif prev_dist - curr_dist < -0.001:
            self.steps_moving_away += 1

        # ── 9. Termination logic ─────────────────────────────────────────────
        is_goal            = (self.current_pos == self.goal_pos)
        is_battery_depleted = (self.battery <= 0.0)

        # Collision terminates only when flag is set
        collision_terminates = is_collision and self.terminate_on_collision

        terminated = is_goal or is_battery_depleted or collision_terminates
        truncated  = (self.step_count >= self.max_steps)

        # Termination reason (exactly one of the six values, or None if continuing)
        if is_goal:
            termination_reason = TERM_GOAL_REACHED
        elif is_battery_depleted:
            termination_reason = TERM_BATTERY_DEPLETED
        elif collision_terminates:
            termination_reason = TERM_COLLISION
        elif truncated:
            termination_reason = TERM_TIMEOUT
        else:
            termination_reason = None

        # ── 10. Reward ───────────────────────────────────────────────────────
        nearest_obstacle_dist = current_cell_data.get("obstacle_distance")
        if nearest_obstacle_dist is None or pd.isnull(nearest_obstacle_dist):
            radar = self.state_builder.get_obstacle_distances(
                self.current_pos, self.mission_gen.grid_free_mask
            )
            nearest_obstacle_dist = float(np.min(radar)) * self.state_builder.max_obstacle_distance

        reward, reward_components = self.reward_calc.calculate_reward(
            prev_dist=prev_dist,
            curr_dist=curr_dist,
            net_score=net_score,
            consecutive_outage_steps=self.consecutive_outage_steps,
            nearest_obstacle_dist=nearest_obstacle_dist,
            energy_spent=step_energy_spent,
            is_collision=is_invalid_move,       # treat invalid move as soft collision
            is_goal=is_goal,
            is_timeout=truncated,
            is_boundary_violation=is_boundary_violation,
        )

        # ── 11. Next observation ─────────────────────────────────────────────
        obs = self._build_obs()

        # ── 12. Info dict ────────────────────────────────────────────────────
        info = {
            # Per-step basics
            "success":              is_goal,
            "is_success":           is_goal,
            "collision":            is_collision,
            "boundary_violation":   is_boundary_violation,
            "invalid_move":         is_invalid_move,
            "battery_depleted":     is_battery_depleted,
            "timeout":              truncated,
            "termination_reason":   termination_reason,
            "step_count":           self.step_count,
            "battery":              self.battery,
            "current_pos":          self.current_pos,
            "distance_to_goal":     curr_dist,
            "network_score":        net_score,
            "latency":              current_cell_data.get("latency"),
            "packet_loss":          current_cell_data.get("packet_loss"),
            "rssi":                 current_cell_data.get("rssi"),
            "consecutive_outage_steps": self.consecutive_outage_steps,
            "total_outage_steps":       self.total_outage_steps,
            "invalid_move_count":       self.invalid_move_count,
            **reward_components,
        }

        # Episode-level aggregates (only on terminal step)
        if terminated or truncated:
            valid_latencies = [l for l in self.episode_latencies    if l is not None and not pd.isnull(l)]
            valid_losses    = [p for p in self.episode_packet_losses if p is not None and not pd.isnull(p)]
            valid_rssis     = [r for r in self.episode_rssi_values   if r is not None and not pd.isnull(r)]

            route_dist = 0.0
            for k in range(len(self.current_route) - 1):
                p1, p2 = self.current_route[k], self.current_route[k + 1]
                route_dist += float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

            info.update({
                "avg_network_score":  float(np.mean(self.episode_network_scores)) if self.episode_network_scores else 0.0,
                "avg_latency":        float(np.mean(valid_latencies)) if valid_latencies else float("nan"),
                "avg_packet_loss":    float(np.mean(valid_losses))    if valid_losses    else float("nan"),
                "avg_rssi":           float(np.mean(valid_rssis))     if valid_rssis     else float("nan"),
                "route_distance":     route_dist,
                "ep_start_pos":       self._ep_start_pos,
                "ep_goal_pos":        self.goal_pos,
                "ep_length":          self.step_count,
                "ep_invalid_move_count": self.invalid_move_count,
                "ep_invalid_move_rate":  self.invalid_move_count / max(1, self.step_count),
                "ep_battery_remaining":  self.battery,
                "ep_distance_to_goal":   curr_dist,
                "ep_distance_reduction": self._ep_start_dist - curr_dist,
                "ep_min_dist_reached":   self.min_dist_reached,
                "ep_steps_closer":       self.steps_moving_closer,
                "ep_steps_away":         self.steps_moving_away,
                "ep_bat_cost_movement":  self.battery_cost_movement,
                "ep_bat_cost_slope":     self.battery_cost_slope,
                "ep_bat_cost_invalid":   self.battery_cost_invalid,
                # Termination flags for callback aggregation
                "term_goal_reached":     is_goal,
                "term_battery_depleted": is_battery_depleted,
                "term_collision":        collision_terminates,
                "term_timeout":          truncated and not (is_goal or is_battery_depleted or collision_terminates),
            })

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        return self.state_builder.build_state(
            current_pos=self.current_pos,
            goal_pos=self.goal_pos,
            battery=self.battery,
            cell_lookup=self.cell_lookup,
            grid_free_mask=self.mission_gen.grid_free_mask,
            feature_processor=self.feature_processor,
            network_quality_calculator=self.net_calc,
        )


if __name__ == "__main__":
    import os
    import sys
    import yaml
    from stable_baselines3.common.env_checker import check_env
    from rl.data.data_loader import load_geo_network_map

    config_path = "rl/config/rl_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    map_path      = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata",        "data/processed/geo_network_metadata.json")

    if not os.path.exists(map_path) or not os.path.exists(metadata_path):
        print("\nERROR: Real Geo-Network Map not found.")
        print(f"Please build {map_path} before training.\n")
        sys.exit(1)

    df, metadata = load_geo_network_map(map_path, metadata_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)

    env = DroneNetworkEnv(df, metadata, config, fp)
    print("Running Gymnasium compliance check...")
    try:
        check_env(env)
        print("Gymnasium compliance check passed!")
    except Exception as e:
        print(f"Compliance check failed: {e}")
        sys.exit(1)
