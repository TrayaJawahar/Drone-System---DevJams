import os
import sys
import yaml
import torch
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from rl.training.seed import set_global_seed
from rl.data.data_loader import load_geo_network_map
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.training.callbacks import CustomMetricsCallback
from rl.evaluation.test_scenarios import generate_and_save_scenarios, load_validation_scenarios
from rl.utils.logger import setup_logger

logger = setup_logger("run_curriculum")

def run_curriculum_training(config_path: str = "rl/config/rl_config.yaml"):
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Check Data
    map_path = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")

    if not os.path.exists(map_path) or not os.path.exists(metadata_path):
        logger.error("Training stopped: missing real Geo-Network Map files.")
        sys.exit(1)

    # 2. Base Setup
    seed = config.get("training", {}).get("seed", 42)
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # 3. Load Map & Feature Processor
    df, metadata = load_geo_network_map(map_path, metadata_path)
    feature_processor = FeatureProcessor()
    feature_processor.fit(df, metadata)

    # 4. Curriculum Loop
    stages = config.get("curriculum", {}).get("stages", [])
    if not stages:
        logger.error("No curriculum stages defined in config.")
        sys.exit(1)

    previous_model_path = None
    previous_stats_path = None

    for stage_idx, stage_config in enumerate(stages):
        stage_name = stage_config.get("name", f"Stage_{stage_idx+1}")
        logger.info(f"\n{'='*50}\nSTARTING CURRICULUM {stage_name}\n{'='*50}")

        # Override config with stage-specific settings
        stage_config_copy = yaml.safe_load(yaml.dump(config)) # deep copy
        stage_config_copy["reward"] = stage_config.get("reward_weights", stage_config_copy.get("reward", {}))
        stage_config_copy["environment"]["max_steps"] = stage_config.get("max_steps", 500)
        
        # In Stage 1, give massive battery
        if stage_idx == 0:
            stage_config_copy["environment"]["battery"]["initial"] = 10000.0

        min_dist = stage_config.get("min_distance", 5)
        max_dist = stage_config.get("max_distance", 20)
        
        adv_criteria = stage_config.get("advancement_criteria", {})
        min_timesteps = adv_criteria.get("min_timesteps", 50000)
        success_threshold = adv_criteria.get("success_rate_threshold", 0.70)

        stage_dir = f"rl/models/stage_{stage_idx+1}"
        os.makedirs(stage_dir, exist_ok=True)
        
        # Update validation scenarios for this stage's distance bounds
        # We temporarily change the PATH in test_scenarios
        import rl.evaluation.test_scenarios as ts
        ts.VALIDATION_PATH = f"rl/evaluation/validation_scenarios_stage_{stage_idx+1}.json"
        
        # Override the generator's random mission bounds via monkey patching DroneNetworkEnv temporarily?
        # Actually, generate_and_save_scenarios uses fixed values. Let's just create custom ones here.
        logger.info(f"Generating validation scenarios for {stage_name} ({min_dist}m - {max_dist}m)")
        from rl.environment.mission_generator import MissionGenerator
        import json
        mission_gen = MissionGenerator(df, metadata)
        val_missions = []
        for _ in range(20):
            try:
                s, g = mission_gen.generate_random_mission(min_distance=min_dist, max_distance=max_dist, max_attempts=5000)
                val_missions.append({"start": [int(s[0]), int(s[1])], "goal": [int(g[0]), int(g[1])]})
            except ValueError:
                pass
        with open(ts.VALIDATION_PATH, "w") as f:
            json.dump(val_missions, f, indent=4)
            
        val_missions_loaded = load_validation_scenarios()

        # Create Environments
        train_raw_env = DroneNetworkEnv(df, metadata, stage_config_copy, feature_processor)
        train_raw_env.set_curriculum_stage(min_dist, max_dist)
        
        # Sanity Check
        if stage_idx == 0:
            check_env(train_raw_env)
            logger.info("Gymnasium compliance check passed for Stage 1.")

        train_mon_env = Monitor(train_raw_env)
        train_vec_env = DummyVecEnv([lambda: train_mon_env])
        train_norm_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_raw_env = DroneNetworkEnv(df, metadata, stage_config_copy, feature_processor)
        eval_raw_env.set_eval_mode(val_missions_loaded)
        eval_mon_env = Monitor(eval_raw_env)
        eval_vec_env = DummyVecEnv([lambda: eval_mon_env])
        eval_norm_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

        # Load previous stage stats if available
        if previous_stats_path and os.path.exists(previous_stats_path):
            logger.info(f"Loading VecNormalize stats from previous stage: {previous_stats_path}")
            train_norm_env = VecNormalize.load(previous_stats_path, train_vec_env)
            train_norm_env.training = True
            train_norm_env.norm_reward = False
        
        eval_norm_env.obs_rms = train_norm_env.obs_rms

        # Setup PPO Model
        ppo_kwargs = {
            "policy": "MlpPolicy",
            "env": train_norm_env,
            "learning_rate": float(config.get("ppo", {}).get("learning_rate", 0.0003)),
            "n_steps": int(config.get("ppo", {}).get("n_steps", 2048)),
            "batch_size": int(config.get("ppo", {}).get("batch_size", 64)),
            "n_epochs": int(config.get("ppo", {}).get("n_epochs", 10)),
            "gamma": float(config.get("ppo", {}).get("gamma", 0.99)),
            "seed": seed,
            "device": device,
            "tensorboard_log": f"rl/logs/stage_{stage_idx+1}",
            "verbose": 1
        }
        
        if previous_model_path and os.path.exists(previous_model_path):
            logger.info(f"Loading PPO model from previous stage: {previous_model_path}")
            model = PPO.load(previous_model_path, env=train_norm_env, device=device)
        else:
            model = PPO(**ppo_kwargs)

        # Setup Callbacks
        metrics_csv = f"rl/logs/stage_{stage_idx+1}_metrics.csv"
        metrics_cb = CustomMetricsCallback(csv_log_path=metrics_csv)
        
        eval_freq = 5000
        eval_cb = EvalCallback(
            eval_env=eval_norm_env,
            n_eval_episodes=len(val_missions_loaded),
            eval_freq=eval_freq,
            best_model_save_path=stage_dir,
            log_path=stage_dir,
            deterministic=True
        )

        all_callbacks = CallbackList([metrics_cb, eval_cb])

        # Train loop
        current_timesteps = 0
        logger.info(f"Training {stage_name} (Target Success: {success_threshold*100}%, Min Steps: {min_timesteps})")
        
        while current_timesteps < 1000000: # hard max cap per stage
            model.learn(total_timesteps=eval_freq, callback=all_callbacks, reset_num_timesteps=False, tb_log_name="PPO")
            current_timesteps += eval_freq
            
            # Check success rate from eval log
            try:
                eval_path = os.path.join(stage_dir, "evaluations.npz")
                if os.path.exists(eval_path):
                    evals = np.load(eval_path)
                    if "successes" in evals:
                        # shape: (num_evals, n_eval_episodes)
                        recent_successes = evals["successes"][-1]
                        goal_rate = np.mean(recent_successes)
                        
                        dist_red = 0.0
                        if len(metrics_cb._window["dist_red"]) > 0:
                            dist_red = np.mean(metrics_cb._window["dist_red"][-100:])
                            
                        logger.info(f"[{current_timesteps} steps] Eval Success Rate: {goal_rate*100:.1f}%, Avg Dist Red: {dist_red:.1f}")
                        
                        if current_timesteps >= min_timesteps and goal_rate >= success_threshold and dist_red > 0:
                            logger.info(f"✅ {stage_name} advancement criteria met!")
                            break
            except Exception as e:
                logger.warning(f"Error reading eval metrics: {e}")

        # Save Final Stage Model
        final_model_path = os.path.join(stage_dir, "model.zip")
        model.save(final_model_path)
        final_stats_path = os.path.join(stage_dir, "vecnormalize.pkl")
        train_norm_env.save(final_stats_path)
        
        previous_model_path = final_model_path
        previous_stats_path = final_stats_path
        logger.info(f"Completed {stage_name}. Model saved.")

    logger.info("All curriculum stages completed successfully.")

if __name__ == "__main__":
    run_curriculum_training()
