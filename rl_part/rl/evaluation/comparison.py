import heapq
import numpy as np
from rl.environment.actions import ACTION_MAP, is_diagonal
from rl.evaluation.metrics import calculate_route_metrics
from rl.utils.logger import setup_logger

logger = setup_logger("comparison")

class AStarPlanner:
    """
    Standard A* search algorithm for an 8-directional grid.
    Optimizes solely for the shortest obstacle-free path, ignoring network quality.
    """
    def __init__(self, grid_free_mask: np.ndarray):
        self.grid_free_mask = grid_free_mask
        self.w, self.h = grid_free_mask.shape

    def _heuristic(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """
        Octile distance heuristic for 8-directional grid movement.
        """
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        # octile distance = (dx + dy) + (sqrt(2) - 2) * min(dx, dy)
        return (dx + dy) + (np.sqrt(2) - 2.0) * min(dx, dy)

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """
        Finds the shortest obstacle-free path from start to goal.
        Returns:
            list of (x, y) coordinates representing the path, or None if no path exists.
        """
        sx, sy = start
        gx, gy = goal
        
        if not self.grid_free_mask[sx, sy] or not self.grid_free_mask[gx, gy]:
            return None

        # Priority queue elements: (f_score, g_score, (x, y))
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0.0, start))
        
        # Parent mapping for reconstruction
        came_from = {}
        
        # Cost from start to current node
        g_score = {start: 0.0}
        
        # Keep track of nodes in open set
        open_set_hash = {start}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current in open_set_hash:
                open_set_hash.remove(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            cx, cy = current
            # Check 8 neighbors
            for action_idx, (dx, dy) in ACTION_MAP.items():
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                # Boundary check
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                    
                # Obstacle check
                if not self.grid_free_mask[nx, ny]:
                    continue

                # Step cost: diagonal is sqrt(2) ~ 1.414, cardinal is 1.0
                step_cost = np.sqrt(2) if is_diagonal(action_idx) else 1.0
                tentative_g = current_g + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        open_set_hash.add(neighbor)

        return None
