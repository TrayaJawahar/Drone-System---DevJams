"""
random_agent.py
---------------
Runs a purely random agent (uniform random actions) for N episodes
and reports episode-level diagnostics to confirm the environment
is healthy before starting PPO training.

Usage:
    python -m rl.diagnostics.random_agent [--episodes 100]

Reports:
  - Average / min / max episode length
  - Average valid action ratio
  - Invalid move rate
  - Timeout rate (episodes that hit max_steps)
  - Success rate (episodes that reached goal)
  - Battery depletion rate
  - Average network score
  - Termination reason breakdown
"""

import os
import sys
import yaml
import argparse
import numpy as np
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.utils.logger import setup_logger

logger = setup_logger("random_agent")


def run_random_agent(n_episodes: int = 100, config_path: str = "rl/config/rl_config.yaml"):
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

    print(f"\n{'='*60}")
    print(f"RANDOM AGENT DIAGNOSTIC — {n_episodes} episodes")
    print(f"terminate_on_collision = {env.terminate_on_collision}")
    print(f"max_steps              = {env.max_steps}")
    print(f"{'='*60}\n")

    ep_lengths       = []
    ep_valid_acts    = []
    ep_invalid_rates = []
    ep_net_scores    = []
    ep_dist_reductions = []
    term_reasons     = Counter()
    successes        = 0
    timeouts         = 0
    battery_deps     = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done      = False
        ep_valid  = 0
        ep_invalid = 0

        while not done:
            action              = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            if step_info.get("invalid_move", False):
                ep_invalid += 1
            else:
                ep_valid += 1

        # Collect episode-level info from final step_info
        ep_len   = int(step_info.get("ep_length", env.step_count))
        inv_rate = float(step_info.get("ep_invalid_move_rate", ep_invalid / max(1, ep_len)))
        avg_net  = float(step_info.get("avg_network_score", 0.0))
        dist_red = float(step_info.get("ep_distance_reduction", 0.0))
        reason   = step_info.get("termination_reason", "timeout")

        ep_lengths.append(ep_len)
        ep_valid_acts.append(ep_valid)
        ep_invalid_rates.append(inv_rate)
        ep_net_scores.append(avg_net)
        ep_dist_reductions.append(dist_red)
        term_reasons[reason] += 1

        if step_info.get("success", False):
            successes += 1
        if step_info.get("timeout", False):
            timeouts += 1
        if step_info.get("battery_depleted", False):
            battery_deps += 1

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:>4}/{n_episodes}  "
                  f"len={ep_len:>4}  "
                  f"inv_rate={inv_rate:.2f}  "
                  f"reason={reason}")

    # ── Report ─────────────────────────────────────────────────────────────
    n = len(ep_lengths)
    print(f"\n{'='*60}")
    print(f"RANDOM AGENT RESULTS  ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Episode length:   mean={np.mean(ep_lengths):.1f}  "
          f"min={min(ep_lengths)}  max={max(ep_lengths)}  "
          f"std={np.std(ep_lengths):.1f}")
    print(f"  Valid actions:    mean={np.mean(ep_valid_acts):.1f}")
    print(f"  Invalid move rate:{np.mean(ep_invalid_rates)*100:.1f}%")
    print(f"  Success rate:     {100*successes/n:.1f}%  ({successes}/{n})")
    print(f"  Timeout rate:     {100*timeouts/n:.1f}%  ({timeouts}/{n})")
    print(f"  Battery depl.:    {100*battery_deps/n:.1f}%  ({battery_deps}/{n})")
    print(f"  Avg network score:{np.mean(ep_net_scores):.4f}")
    print(f"  Avg dist reduction:{np.mean(ep_dist_reductions):.2f} cells")
    print(f"\n  Termination breakdown:")
    for reason, cnt in sorted(term_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25}: {cnt:>4}  ({100*cnt/n:.1f}%)")

    # ── Verdict ─────────────────────────────────────────────────────────────
    avg_len = np.mean(ep_lengths)
    print(f"\n{'='*60}")
    print(f"VERDICT:")
    if avg_len > 10:
        print(f"  ✅ Average episode length {avg_len:.1f} > 10 — environment is healthy.")
        print(f"  PPO training can proceed.")
    else:
        print(f"  ❌ Average episode length {avg_len:.1f} ≤ 10 — environment still broken.")
        print(f"  Do NOT start PPO training yet. Check termination reasons above.")
    print("=" * 60 + "\n")

    return {
        "avg_ep_length":    float(np.mean(ep_lengths)),
        "avg_invalid_rate": float(np.mean(ep_invalid_rates)),
        "success_rate":     float(successes / n),
        "timeout_rate":     float(timeouts / n),
        "battery_rate":     float(battery_deps / n),
        "avg_net_score":    float(np.mean(ep_net_scores)),
        "term_reasons":     dict(term_reasons),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", default="rl/config/rl_config.yaml")
    args = parser.parse_args()
    run_random_agent(args.episodes, args.config)
