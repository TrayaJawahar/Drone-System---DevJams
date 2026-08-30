import os
import sys
import yaml
import pandas as pd
from rl.training.train_ppo import run_training
from rl.evaluation.evaluate import run_evaluation
from rl.evaluation.test_scenarios import load_test_scenarios
from rl.inference.route_inference import RouteInferenceEngine
from rl.evaluation.comparison import AStarPlanner
from rl.utils.visualization import plot_network_heatmap, compare_routes
from rl.utils.logger import setup_logger

logger = setup_logger("run_training_entrypoint")

def main():
    config_path = "rl/config/rl_config.yaml"
    
    # 1. Load config to check data path
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    map_path = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")

    # 2. Enforce Critical Data Policy
    if not os.path.exists(map_path) or not os.path.exists(metadata_path):
        print("\nERROR: Real Geo-Network Map not found.")
        print(f"Please build {map_path} before training.\n")
        logger.error(f"Execution halted. Real map files missing. Check: {map_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STARTING REINFORCEMENT LEARNING TRAINING SYSTEM")
    logger.info("=" * 60)

    # 3. Run PPO Training Subsystem
    try:
        run_training(config_path)
    except Exception as e:
        logger.critical(f"Training failed: {e}")
        sys.exit(1)

    # 4. Run Final PPO vs A* Evaluation
    logger.info("Running evaluation on unseen test scenarios...")
    try:
        run_evaluation()
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}")
        sys.exit(1)

    # 5. Generate Visual Demonstrations
    logger.info("Generating route planning visualization...")
    try:
        # Load a test scenario and run inference to plot comparison
        test_scenarios = load_test_scenarios()
        if test_scenarios:
            start, goal = test_scenarios[0]
            
            # Instantiate inference engine to get routes
            engine = RouteInferenceEngine(config_path=config_path)
            
            # Plot general network quality heatmap
            plot_network_heatmap(engine.df, save_path="rl/logs/network_heatmap.png")
            
            # Predict PPO route
            ppo_res = engine.generate_route(start, goal)
            ppo_route = ppo_res["route_cells"] if ppo_res["success"] else []
            
            # Plan A* route
            astar = AStarPlanner(engine.raw_env.mission_gen.grid_free_mask)
            astar_route = astar.plan(start, goal)
            
            # Plot comparison plot
            compare_routes(
                df=engine.df,
                ppo_route=ppo_route,
                astar_route=astar_route,
                start=start,
                goal=goal,
                save_path="rl/logs/route_comparison.png"
            )
            logger.info("Route planning visualization figures generated successfully in rl/logs/")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")

    logger.info("=" * 60)
    logger.info("RL TRAINING SYSTEM PIPELINE FINISHED SUCCESSFULLY")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
