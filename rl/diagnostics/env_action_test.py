"""
env_action_test.py
------------------
Manual environment test: executes all 8 actions from a known free cell
and verifies position updates, obstacle detection, boundary detection,
reward values, and termination status.

Usage:
    python -m rl.diagnostics.env_action_test
"""

import os
import sys
import yaml
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.environment.actions import ACTION_NAMES
from rl.utils.logger import setup_logger

logger = setup_logger("env_action_test")


def run_action_test(config_path: str = "rl/config/rl_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    map_path  = config["data"]["geo_network_map"]
    meta_path = config["data"]["metadata"]

    if not os.path.exists(map_path):
        print(f"\nERROR: Real Geo-Network Map not found at {map_path}\n")
        sys.exit(1)

    df, metadata = load_geo_network_map(map_path, meta_path)
    fp = FeatureProcessor()
    fp.fit(df, metadata)

    env = DroneNetworkEnv(df, metadata, config, fp)

    print("\n" + "=" * 70)
    print("ENVIRONMENT ACTION TEST")
    print("=" * 70)

    # ── Find a good test cell ────────────────────────────────────────────────
    # Pick the first good start cell (has >= 2 free neighbors)
    good_starts = env.mission_gen._good_start_cells
    if not good_starts:
        print("ERROR: No good start cells found!")
        sys.exit(1)

    # Find a start cell that has at least one action leading to an obstacle
    # and at least one leading out of bounds to test both invalid-move cases
    chosen_start = good_starts[0]
    w = env.grid_width
    h = env.grid_height

    # Try to find a cell near a corner (likely to have boundary violations)
    corner_cell = None
    for pos in good_starts:
        gx, gy = pos
        if gx <= 2 or gy <= 2 or gx >= w - 3 or gy >= h - 3:
            corner_cell = pos
            break
    test_cell = corner_cell if corner_cell else chosen_start

    print(f"\nTest cell: {test_cell}  Grid: {w}x{h}")
    print(f"Cell is free: {env.mission_gen.grid_free_mask[test_cell[0], test_cell[1]]}")

    # Count free neighbors
    from rl.environment.actions import ACTION_MAP
    free_n = sum(
        1 for _, (dx, dy) in ACTION_MAP.items()
        if 0 <= test_cell[0]+dx < w and 0 <= test_cell[1]+dy < h
        and env.mission_gen.grid_free_mask[test_cell[0]+dx, test_cell[1]+dy]
    )
    print(f"Free neighbors: {free_n} / 8")

    # ── Reset env to known start, arbitrary goal ─────────────────────────────
    # Use a reachable goal far enough away
    import random
    random.seed(42)
    free_list = list(env.mission_gen.free_cells)
    goal = None
    for _ in range(1000):
        cand = random.choice(free_list)
        dist = np.sqrt((test_cell[0]-cand[0])**2 + (test_cell[1]-cand[1])**2)
        if dist >= 5 and env.mission_gen.is_reachable(test_cell, cand):
            goal = cand
            break
    if goal is None:
        goal = free_list[0]

    obs, info = env.reset(options={"start": test_cell, "goal": goal})
    print(f"Goal cell: {goal}")
    print(f"Start dist to goal: {np.sqrt((test_cell[0]-goal[0])**2+(test_cell[1]-goal[1])**2):.1f} cells")
    print(f"Obs shape: {obs.shape}")

    print("\n" + "-" * 70)
    print(f"{'Action':<14} {'Name':<14} {'New Pos':<12} {'Valid?':<7} {'Boundary?':<11} {'Collision?':<11} {'Reward':>8} {'Term?':<6}")
    print("-" * 70)

    results = []
    for action in range(8):
        # Reset to test cell before each action
        env.reset(options={"start": test_cell, "goal": goal})
        obs, reward, terminated, truncated, step_info = env.step(action)

        new_pos         = step_info["current_pos"]
        is_boundary     = step_info["boundary_violation"]
        is_coll         = step_info["collision"]
        is_invalid      = step_info["invalid_move"]
        moved           = new_pos != test_cell
        term_reason     = step_info.get("termination_reason", "-")

        action_name = ACTION_NAMES[action]
        dx, dy      = ACTION_MAP[action]
        expected_nx = test_cell[0] + dx
        expected_ny = test_cell[1] + dy

        in_bounds = 0 <= expected_nx < w and 0 <= expected_ny < h
        if in_bounds:
            is_free = env.mission_gen.grid_free_mask[expected_nx, expected_ny]
        else:
            is_free = False

        expected_new_pos = (expected_nx, expected_ny) if (in_bounds and is_free) else test_cell
        pos_correct = new_pos == expected_new_pos

        status = "✅" if pos_correct else "❌"
        results.append({
            "action": action,
            "name": action_name,
            "new_pos": new_pos,
            "moved": moved,
            "is_boundary": is_boundary,
            "is_collision": is_coll,
            "is_invalid": is_invalid,
            "reward": reward,
            "terminated": terminated or truncated,
            "term_reason": term_reason,
            "pos_correct": pos_correct,
        })

        print(
            f"{status} {action_name:<13} {str(new_pos):<12} {'YES' if not is_invalid else 'no':<7} "
            f"{'YES' if is_boundary else '-':<11} {'YES' if is_coll else '-':<11} "
            f"{reward:>8.3f} {'YES' if terminated or truncated else '-':<6}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    valid_moves   = sum(1 for r in results if not r["is_invalid"])
    invalid_moves = sum(1 for r in results if r["is_invalid"])
    boundary_moves= sum(1 for r in results if r["is_boundary"])
    coll_moves    = sum(1 for r in results if r["is_collision"])
    pos_correct   = sum(1 for r in results if r["pos_correct"])

    print("-" * 70)
    print(f"\nSummary for cell {test_cell}:")
    print(f"  Valid moves    : {valid_moves} / 8")
    print(f"  Invalid moves  : {invalid_moves} / 8  "
          f"({boundary_moves} boundary, {coll_moves} obstacle)")
    print(f"  Pos correct    : {pos_correct} / 8")
    print(f"  No premature termination: {'✅' if all(not r['terminated'] for r in results) else '⚠️  Some actions terminated early!'}")
    print(f"\n  Rewards range: [{min(r['reward'] for r in results):.3f}, "
          f"{max(r['reward'] for r in results):.3f}]")

    print("\n" + "=" * 70)
    print("All 8 actions tested. Position stays on invalid moves: "
          + ("✅ CONFIRMED" if all(
              r["new_pos"] == test_cell for r in results if r["is_invalid"]
          ) else "❌ FAILED"))
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_action_test()
