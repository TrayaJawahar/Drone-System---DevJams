"""
mission_generator.py
--------------------
Generates reachable start-goal mission configurations on the grid.

Key guarantees:
- Start and goal are both free (non-obstacle) cells.
- Start has at least `min_start_neighbors` free neighboring cells so the
  drone is not immediately trapped with no valid moves.
- Goal is reachable from start via 8-connectivity (checked in O(1) using
  pre-computed connected components).
- Euclidean distance between start and goal is within curriculum bounds.
"""

import random
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from rl.environment.actions import ACTION_MAP
from rl.utils.logger import setup_logger

logger = setup_logger("mission_generator")

# Minimum number of free neighbors start cell must have
MIN_START_NEIGHBORS = 2


class MissionGenerator:
    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        self.df       = df
        self.metadata = metadata

        self.max_x      = int(df["grid_x"].max())
        self.max_y      = int(df["grid_y"].max())
        self.grid_width  = self.max_x + 1
        self.grid_height = self.max_y + 1

        # Build free-cell set and mask
        self.free_cells     = set()
        self.grid_free_mask = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        for _, row in df.iterrows():
            gx    = int(row["grid_x"])
            gy    = int(row["grid_y"])
            is_obs = bool(row["is_obstacle"])
            if not is_obs:
                self.free_cells.add((gx, gy))
                self.grid_free_mask[gx, gy] = True

        # Pre-compute 8-connected components for O(1) reachability checks
        connectivity = np.ones((3, 3), dtype=int)
        self.labeled_grid, self.num_components = ndimage.label(
            self.grid_free_mask, structure=connectivity
        )
        logger.info(
            f"MissionGenerator ready: {self.grid_width}x{self.grid_height} grid, "
            f"{len(self.free_cells)} free cells, {self.num_components} connected component(s)."
        )

        # Pre-filter: free cells with at least MIN_START_NEIGHBORS free neighbors
        self._good_start_cells = [
            pos for pos in self.free_cells
            if self._count_free_neighbors(pos) >= MIN_START_NEIGHBORS
        ]
        logger.info(
            f"Good start cells (>={MIN_START_NEIGHBORS} free neighbors): "
            f"{len(self._good_start_cells)}"
        )
        if not self._good_start_cells:
            logger.warning(
                "No cells found with enough free neighbors! "
                "Falling back to all free cells as potential starts."
            )
            self._good_start_cells = list(self.free_cells)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_reachable(self, start: tuple, goal: tuple) -> bool:
        """O(1) reachability check via connected-component labels."""
        sx, sy = start
        gx, gy = goal

        if not (0 <= sx < self.grid_width  and 0 <= sy < self.grid_height):
            return False
        if not (0 <= gx < self.grid_width  and 0 <= gy < self.grid_height):
            return False
        if not self.grid_free_mask[sx, sy] or not self.grid_free_mask[gx, gy]:
            return False

        sl = self.labeled_grid[sx, sy]
        gl = self.labeled_grid[gx, gy]
        return sl == gl and sl > 0

    def validate_mission(self, start: tuple, goal: tuple) -> tuple[bool, str]:
        """
        Full mission validation.

        Returns:
            (valid: bool, reason: str)
        """
        sx, sy = start
        gx, gy = goal

        if not self.grid_free_mask[sx, sy]:
            return False, f"Start {start} is an obstacle."
        if not self.grid_free_mask[gx, gy]:
            return False, f"Goal {goal} is an obstacle."
        if not self.is_reachable(start, goal):
            return False, f"Goal {goal} unreachable from start {start}."
        n_neighbors = self._count_free_neighbors(start)
        if n_neighbors < MIN_START_NEIGHBORS:
            return False, f"Start {start} has only {n_neighbors} free neighbor(s); min={MIN_START_NEIGHBORS}."
        return True, "ok"

    def generate_random_mission(
        self,
        min_distance: float = 10.0,
        max_distance: float = 200.0,
        max_attempts: int   = 2000,
    ) -> tuple[tuple, tuple]:
        """
        Generates a valid (free, reachable, well-separated) start-goal pair.

        The start is drawn from the pre-filtered list of cells with enough
        free neighbors.  The goal is drawn from all free cells.
        """
        free_list = list(self.free_cells)
        if len(free_list) < 2:
            raise ValueError("Not enough free cells to generate a mission.")

        for _ in range(max_attempts):
            start = random.choice(self._good_start_cells)
            goal  = random.choice(free_list)

            if start == goal:
                continue

            dist = float(np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
            if min_distance <= dist <= max_distance:
                if self.is_reachable(start, goal):
                    return start, goal

        # Fallback: any reachable pair from good-start cells
        logger.warning(
            f"Could not find mission in [{min_distance}, {max_distance}] after "
            f"{max_attempts} attempts. Selecting any reachable pair."
        )
        for _ in range(max_attempts):
            start = random.choice(self._good_start_cells)
            goal  = random.choice(free_list)
            if start != goal and self.is_reachable(start, goal):
                return start, goal

        raise ValueError("Could not generate any reachable start-goal mission.")

    def generate_fixed_mission(self, start: tuple, goal: tuple) -> tuple[tuple, tuple]:
        """Validates and returns a fixed mission, raising ValueError on failure."""
        valid, reason = self.validate_mission(start, goal)
        if not valid:
            raise ValueError(f"Invalid fixed mission: {reason}")
        return start, goal

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _count_free_neighbors(self, pos: tuple) -> int:
        """Counts how many of the 8 neighbors of `pos` are free (non-obstacle, in-bounds)."""
        cx, cy  = pos
        count   = 0
        for _, (dx, dy) in ACTION_MAP.items():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                if self.grid_free_mask[nx, ny]:
                    count += 1
        return count
