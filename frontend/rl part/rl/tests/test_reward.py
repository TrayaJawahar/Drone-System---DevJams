import pytest
from rl.environment.reward import RewardCalculator

# Configuration for testing reward calculator
REWARD_CONFIG = {
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

def test_reward_components():
    rc = RewardCalculator(REWARD_CONFIG)
    
    # 1. Moving closer vs moving away
    # Moving closer by 2 grid units
    r_close, comp_close = rc.calculate_reward(
        prev_dist=10.0, curr_dist=8.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    # Moving away by 1 grid unit
    r_away, comp_away = rc.calculate_reward(
        prev_dist=10.0, curr_dist=11.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    
    assert comp_close["progress_reward"] > 0.0
    assert comp_away["progress_reward"] < 0.0
    assert r_close > r_away

    # 2. Strong network vs weak network
    # Strong network score = 0.9
    _, comp_strong = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.9,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    # Weak network score = 0.4 (no outage yet, threshold is 0.3)
    _, comp_weak = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.4,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    
    assert comp_strong["network_reward"] > comp_weak["network_reward"]
    assert comp_strong["outage_penalty"] == 0.0
    assert comp_weak["outage_penalty"] == 0.0

    # 3. Connectivity Outage duration scaling
    # Outage step 1 (network score = 0.2 < threshold 0.3)
    _, comp_outage_1 = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.2,
        consecutive_outage_steps=1, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    # Outage step 5
    _, comp_outage_5 = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.2,
        consecutive_outage_steps=5, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    
    # Outage penalty is negative, check absolute value
    assert comp_outage_1["outage_penalty"] < 0.0
    assert comp_outage_5["outage_penalty"] < comp_outage_1["outage_penalty"]  # More negative for consecutive steps

    # 4. Proximity safety penalty
    # Far from obstacles (dist 5.0 >= safety threshold 3.0)
    _, comp_safe = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    # Near obstacle (dist 1.0 < safety threshold 3.0)
    _, comp_near = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=1.0,
        energy_spent=0.5, is_collision=False, is_goal=False, is_timeout=False
    )
    assert comp_safe["safety_reward"] == 0.0
    assert comp_near["safety_reward"] < 0.0

    # 5. Goal reward, Collision penalty
    _, comp_goal = rc.calculate_reward(
        prev_dist=10.0, curr_dist=0.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=False, is_goal=True, is_timeout=False
    )
    _, comp_coll = rc.calculate_reward(
        prev_dist=10.0, curr_dist=10.0, net_score=0.8,
        consecutive_outage_steps=0, nearest_obstacle_dist=5.0,
        energy_spent=0.5, is_collision=True, is_goal=False, is_timeout=False
    )
    
    assert comp_goal["goal_reward"] == 100.0
    assert comp_coll["collision_penalty"] == -100.0
