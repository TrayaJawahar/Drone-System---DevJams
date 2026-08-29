import os
import json
from rl.environment.mission_generator import MissionGenerator
from rl.utils.logger import setup_logger

logger = setup_logger("test_scenarios")

VALIDATION_PATH = "rl/evaluation/validation_scenarios.json"
TEST_PATH = "rl/evaluation/test_scenarios.json"

def generate_and_save_scenarios(df, metadata, num_scenarios=20, force=False):
    """
    Generates and saves fixed validation and test mission scenarios if they do not exist.
    """
    os.makedirs(os.path.dirname(VALIDATION_PATH), exist_ok=True)
    
    mission_gen = MissionGenerator(df, metadata)
    
    # Generate Validation Scenarios
    if not os.path.exists(VALIDATION_PATH) or force:
        logger.info(f"Generating {num_scenarios} fixed validation missions...")
        val_missions = []
        for i in range(num_scenarios):
            # Mid to long distance validation
            start, goal = mission_gen.generate_random_mission(min_distance=20.0, max_distance=100.0)
            val_missions.append({
                "start": [int(start[0]), int(start[1])],
                "goal": [int(goal[0]), int(goal[1])]
            })
        with open(VALIDATION_PATH, "w") as f:
            json.dump(val_missions, f, indent=4)
        logger.info(f"Saved validation scenarios to {VALIDATION_PATH}")
    else:
        logger.info(f"Validation scenarios already exist at {VALIDATION_PATH}")

    # Generate Test Scenarios
    if not os.path.exists(TEST_PATH) or force:
        logger.info(f"Generating {num_scenarios} fixed test missions...")
        test_missions = []
        for i in range(num_scenarios):
            # Mid to long distance test
            start, goal = mission_gen.generate_random_mission(min_distance=20.0, max_distance=100.0)
            test_missions.append({
                "start": [int(start[0]), int(start[1])],
                "goal": [int(goal[0]), int(goal[1])]
            })
        with open(TEST_PATH, "w") as f:
            json.dump(test_missions, f, indent=4)
        logger.info(f"Saved test scenarios to {TEST_PATH}")
    else:
        logger.info(f"Test scenarios already exist at {TEST_PATH}")

def load_validation_scenarios() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Loads fixed validation scenarios.
    """
    if not os.path.exists(VALIDATION_PATH):
        raise FileNotFoundError(f"Validation scenarios file not found at {VALIDATION_PATH}")
    with open(VALIDATION_PATH, "r") as f:
        data = json.load(f)
    return [((m["start"][0], m["start"][1]), (m["goal"][0], m["goal"][1])) for m in data]

def load_test_scenarios() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Loads fixed test scenarios.
    """
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test scenarios file not found at {TEST_PATH}")
    with open(TEST_PATH, "r") as f:
        data = json.load(f)
    return [((m["start"][0], m["start"][1]), (m["goal"][0], m["goal"][1])) for m in data]
