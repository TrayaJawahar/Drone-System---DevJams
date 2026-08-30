from rl.utils.logger import setup_logger

logger = setup_logger("reward")

class RewardCalculator:
    """
    Computes separate, configurable reward components for the RL environment.
    Balances safety, route distance, battery capacity, and network availability.

    Invalid moves (obstacle collision / boundary violation) are NOT terminating
    by default.  They receive a configurable penalty so PPO can learn to avoid
    them without the episode ending.
    """
    def __init__(self, config: dict):
        self.config         = config
        self.reward_config  = config.get("reward", {})

        # Weights
        self.w_progress   = float(self.reward_config.get("progress_weight",    5.0))
        self.w_network    = float(self.reward_config.get("network_weight",     3.0))
        self.w_safety     = float(self.reward_config.get("safety_weight",      2.0))

        # Step costs
        self.penalty_movement  = float(self.reward_config.get("movement_penalty",   0.1))
        self.penalty_energy    = float(self.reward_config.get("energy_penalty",     0.05))
        self.penalty_outage    = float(self.reward_config.get("outage_penalty",     5.0))

        # Invalid move penalty (non-terminating collision / boundary violation)
        self.penalty_invalid   = float(self.reward_config.get("invalid_move_penalty", 2.0))

        # Terminal penalties
        self.penalty_collision = float(self.reward_config.get("collision_penalty",  100.0))  # only when terminate_on_collision=True
        self.penalty_timeout   = float(self.reward_config.get("timeout_penalty",    10.0))

        self.reward_goal = float(self.reward_config.get("goal_reward", 100.0))

        # Network outage threshold
        net_config             = config.get("network", {})
        self.outage_threshold  = float(net_config.get("outage_threshold", 0.3))

        # Safety proximity threshold (grid units)
        self.safety_threshold  = 3.0

    def calculate_reward(
        self,
        prev_dist:               float,
        curr_dist:               float,
        net_score:               float,
        consecutive_outage_steps: int,
        nearest_obstacle_dist:   float,
        energy_spent:            float,
        is_collision:            bool,   # True = drone tried to enter obstacle / boundary
        is_goal:                 bool,
        is_timeout:              bool,
        is_boundary_violation:   bool = False,
    ) -> tuple[float, dict]:
        """
        Calculates the aggregate step reward and breaks down individual components.

        is_collision is True for ANY invalid move (obstacle OR boundary).
        The penalty applied is penalty_invalid (not penalty_collision) unless
        the episode is explicitly terminated by a collision.

        Returns:
            total_reward:       float
            reward_components:  dict
        """
        # 1. Goal Progress reward
        #    Moving closer → positive; moving away or staying → negative / zero
        progress        = prev_dist - curr_dist
        progress_reward = progress * self.w_progress

        # 2. Network Quality reward
        network_reward  = net_score * self.w_network

        # 3. Outage penalty (scales with consecutive outage steps)
        outage_penalty = 0.0
        if net_score < self.outage_threshold:
            outage_penalty = self.penalty_outage * (1.0 + 0.2 * (consecutive_outage_steps - 1))

        # 4. Safety Proximity penalty
        safety_reward = 0.0
        if nearest_obstacle_dist < self.safety_threshold:
            proximity_factor = (self.safety_threshold - nearest_obstacle_dist) / self.safety_threshold
            safety_reward    = -proximity_factor * self.w_safety

        # 5. Movement cost (prevents infinite wandering)
        movement_cost = -self.penalty_movement

        # 6. Energy cost
        energy_cost = -energy_spent * self.penalty_energy

        # 7. Invalid-move penalty (non-terminating)
        #    Applied to both boundary violations and obstacle collisions.
        invalid_penalty = -self.penalty_invalid if is_collision else 0.0

        # 8. Terminal bonuses / penalties
        goal_bonus         = self.reward_goal if is_goal else 0.0
        timeout_penalty    = -self.penalty_timeout if is_timeout else 0.0

        # Sum
        total_reward = (
            progress_reward
            + network_reward
            + safety_reward
            + movement_cost
            + energy_cost
            + invalid_penalty
            + goal_bonus
            + timeout_penalty
            - outage_penalty
        )

        components = {
            "progress_reward":   float(progress_reward),
            "network_reward":    float(network_reward),
            "safety_reward":     float(safety_reward),
            "outage_penalty":    float(-outage_penalty),
            "movement_penalty":  float(movement_cost),
            "energy_penalty":    float(energy_cost),
            "invalid_penalty":   float(invalid_penalty),
            "goal_reward":       float(goal_bonus),
            "timeout_penalty":   float(timeout_penalty),
            "total_reward":      float(total_reward),
        }

        return float(total_reward), components
