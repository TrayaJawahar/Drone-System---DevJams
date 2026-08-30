import numpy as np

# Action mapping: Action Index -> (delta_x, delta_y)
ACTION_MAP = {
    0: (0, 1),    # NORTH
    1: (0, -1),   # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (1, 1),    # NORTH_EAST
    5: (-1, 1),   # NORTH_WEST
    6: (1, -1),   # SOUTH_EAST
    7: (-1, -1)   # SOUTH_WEST
}

ACTION_NAMES = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "NORTH_EAST",
    5: "NORTH_WEST",
    6: "SOUTH_EAST",
    7: "SOUTH_WEST"
}

def get_movement(action: int) -> tuple[int, int]:
    """
    Returns the (dx, dy) grid movement corresponding to a given action.
    """
    if action not in ACTION_MAP:
        raise ValueError(f"Invalid action: {action}. Action must be in range 0-7.")
    return ACTION_MAP[action]

def is_diagonal(action: int) -> bool:
    """
    Returns True if the action is diagonal (actions 4-7).
    """
    return action >= 4
