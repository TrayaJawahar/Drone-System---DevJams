"""
callbacks.py
------------
Custom Stable-Baselines3 callbacks for the Drone RL training pipeline.

CustomMetricsCallback
  - Logs per-episode metrics to TensorBoard and CSV.
  - Tracks all 6 termination reasons separately.
  - Logs episode length, invalid move rate, distance-to-goal reduction.

CurriculumCallback
  - Advances curriculum distance stages based on training progress.
"""

import os
import csv
import numpy as np
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from rl.utils.logger import setup_logger

logger = setup_logger("callbacks")


class CustomMetricsCallback(BaseCallback):
    """
    Logs rich episode-level statistics to TensorBoard and a CSV file.

    Per-episode logging:
      - episode number, step, reward
      - termination reason (goal / battery / collision / timeout)
      - episode length, invalid move rate
      - distance to goal, distance reduction from start
      - battery remaining
      - avg network score, avg RSSI, avg latency, avg packet loss
      - outage steps

    Aggregated TensorBoard metrics (rolling window):
      - average episode length
      - termination reason counts / rates
      - goal reached count, invalid move rate
      - mean distance reduction per episode
    """

    def __init__(self, csv_log_path: str = "logs/training_metrics.csv", verbose: int = 0):
        super().__init__(verbose)
        self.csv_log_path = csv_log_path
        self.episode_count = 0

        # Rolling window accumulators (reset each time we dump to TensorBoard)
        self._window: dict[str, list] = defaultdict(list)

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
        try:
            with open(self.csv_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "global_step",
                    "ep_reward",
                    "ep_length",
                    "termination_reason",
                    "term_goal",
                    "term_battery",
                    "term_collision",
                    "term_timeout",
                    "start_x", "start_y",
                    "goal_x",  "goal_y",
                    "distance_to_goal",
                    "distance_reduction",
                    "invalid_move_count",
                    "invalid_move_rate",
                    "battery_remaining",
                    "avg_network_score",
                    "avg_rssi",
                    "avg_latency",
                    "avg_packet_loss",
                    "total_outage_steps",
                    "route_distance",
                ])
            logger.info(f"Initialized metrics CSV at {self.csv_log_path}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}")

    def _on_step(self) -> bool:
        for idx, info in enumerate(self.locals.get("infos", [])):
            if not self.locals.get("dones")[idx]:
                continue

            self.episode_count += 1

            # ── Pull episode-level fields from terminal info ───────────────
            term_reason  = info.get("termination_reason", "unknown")
            term_goal    = bool(info.get("term_goal_reached",     False))
            term_battery = bool(info.get("term_battery_depleted", False))
            term_coll    = bool(info.get("term_collision",        False))
            term_timeout = bool(info.get("term_timeout",          True))  # default to timeout if unknown

            ep_len        = int(info.get("ep_length",            info.get("step_count", 0)))
            inv_count     = int(info.get("ep_invalid_move_count", 0))
            inv_rate      = float(info.get("ep_invalid_move_rate", 0.0))
            battery_rem   = float(info.get("ep_battery_remaining", info.get("battery", 0.0)))
            dist_goal     = float(info.get("ep_distance_to_goal",  info.get("distance_to_goal", -1.0)))
            dist_red      = float(info.get("ep_distance_reduction", 0.0))

            start_pos = info.get("ep_start_pos", (-1, -1))
            goal_pos  = info.get("ep_goal_pos",  (-1, -1))

            avg_net   = float(info.get("avg_network_score", 0.0))
            avg_rssi  = info.get("avg_rssi",        float("nan"))
            avg_lat   = info.get("avg_latency",     float("nan"))
            avg_loss  = info.get("avg_packet_loss", float("nan"))
            outage_s  = int(info.get("total_outage_steps", 0))
            route_d   = float(info.get("route_distance", 0.0))

            # Episode reward: prefer Monitor wrapper value
            ep_reward = 0.0
            if "episode" in info:
                ep_reward = float(info["episode"].get("r", 0.0))
            else:
                ep_reward = float(info.get("total_reward", 0.0))

            # ── TensorBoard (per-episode scalars) ─────────────────────────
            self.logger.record("episode/length",              ep_len)
            self.logger.record("episode/reward",              ep_reward)
            self.logger.record("episode/invalid_move_rate",   inv_rate)
            self.logger.record("episode/distance_to_goal",    dist_goal)
            self.logger.record("episode/distance_reduction",  dist_red)
            self.logger.record("episode/battery_remaining",   battery_rem)

            # Termination: log 0/1 per reason so TB shows separate curves
            self.logger.record("termination/goal_reached",     1.0 if term_goal    else 0.0)
            self.logger.record("termination/battery_depleted", 1.0 if term_battery else 0.0)
            self.logger.record("termination/collision",        1.0 if term_coll    else 0.0)
            self.logger.record("termination/timeout",          1.0 if term_timeout else 0.0)

            self.logger.record("network/mean_quality",    avg_net)
            self.logger.record("network/outage_steps",    outage_s)

            if not np.isnan(avg_rssi):
                self.logger.record("network/mean_rssi",         avg_rssi)
            if not np.isnan(avg_lat):
                self.logger.record("network/mean_latency",      avg_lat)
            if not np.isnan(avg_loss):
                self.logger.record("network/mean_packet_loss",  avg_loss)

            # Rolling window for aggregated rates
            self._window["ep_len"].append(ep_len)
            self._window["goal"].append(1 if term_goal else 0)
            self._window["battery"].append(1 if term_battery else 0)
            self._window["collision"].append(1 if term_coll else 0)
            self._window["timeout"].append(1 if term_timeout else 0)
            self._window["inv_rate"].append(inv_rate)
            self._window["dist_red"].append(dist_red)
            self._window["net"].append(avg_net)

            W = 100  # window size
            if len(self._window["ep_len"]) >= W:
                self.logger.record("agg/avg_ep_len",        np.mean(self._window["ep_len"][-W:]))
                self.logger.record("agg/goal_rate",         np.mean(self._window["goal"][-W:]))
                self.logger.record("agg/battery_rate",      np.mean(self._window["battery"][-W:]))
                self.logger.record("agg/collision_rate",    np.mean(self._window["collision"][-W:]))
                self.logger.record("agg/timeout_rate",      np.mean(self._window["timeout"][-W:]))
                self.logger.record("agg/invalid_move_rate", np.mean(self._window["inv_rate"][-W:]))
                self.logger.record("agg/dist_reduction",    np.mean(self._window["dist_red"][-W:]))
                self.logger.record("agg/net_quality",       np.mean(self._window["net"][-W:]))

            # ── CSV row ───────────────────────────────────────────────────
            try:
                with open(self.csv_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_count,
                        self.num_timesteps,
                        ep_reward,
                        ep_len,
                        term_reason,
                        int(term_goal),
                        int(term_battery),
                        int(term_coll),
                        int(term_timeout),
                        start_pos[0], start_pos[1],
                        goal_pos[0],  goal_pos[1],
                        dist_goal,
                        dist_red,
                        inv_count,
                        inv_rate,
                        battery_rem,
                        avg_net,
                        avg_rssi if not (isinstance(avg_rssi, float) and np.isnan(avg_rssi)) else "",
                        avg_lat  if not (isinstance(avg_lat,  float) and np.isnan(avg_lat))  else "",
                        avg_loss if not (isinstance(avg_loss, float) and np.isnan(avg_loss)) else "",
                        outage_s,
                        route_d,
                    ])
            except Exception as e:
                logger.error(f"Error writing metrics CSV: {e}")

        return True


class CurriculumCallback(BaseCallback):
    """
    Advances curriculum distance stages based on total training timesteps.
    """

    def __init__(self, train_env, stages: list, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.train_env       = train_env
        self.stages          = stages
        self.total_timesteps = total_timesteps
        self.num_stages      = len(stages)
        self.stage_duration  = total_timesteps / max(1, self.num_stages)
        self.current_stage   = -1

    def _on_step(self) -> bool:
        stage_idx = min(int(self.num_timesteps / self.stage_duration), self.num_stages - 1)

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            cfg   = self.stages[stage_idx]
            min_d = cfg.get("min_distance", 10.0)
            max_d = cfg.get("max_distance", 30.0)
            try:
                self.train_env.env_method("set_curriculum_stage", min_d, max_d)
                logger.info(
                    f"[Curriculum] Stage {stage_idx+1}/{self.num_stages}: "
                    f"dist=[{min_d}, {max_d}] at step {self.num_timesteps}"
                )
            except Exception as e:
                logger.error(f"Failed to set curriculum stage: {e}")

        return True
