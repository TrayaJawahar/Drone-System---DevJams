import os
import sys
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

from rl.training.seed import set_global_seed
from rl.data.data_loader import load_geo_network_map, get_data_quality_summary, print_quality_summary_report
from rl.data.feature_processor import FeatureProcessor
from rl.environment.drone_network_env import DroneNetworkEnv
from rl.training.callbacks import CustomMetricsCallback, CurriculumCallback
from rl.evaluation.test_scenarios import generate_and_save_scenarios, load_validation_scenarios
from rl.utils.logger import setup_logger

logger = setup_logger("train_ppo")

class SaveVecNormalizeOnBestCallback(BaseCallback):
    """
    Callback that saves the VecNormalize statistics whenever a new best model is found.
    """
    def __init__(self, vecnormalize_env: VecNormalize, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.vecnormalize_env = vecnormalize_env
        self.save_dir = save_dir
        
    def _on_step(self) -> bool:
        stats_path = os.path.join(self.save_dir, "best_model_vecnormalize.pkl")
        self.vecnormalize_env.save(stats_path)
        logger.info(f"[Callback] Saved best VecNormalize statistics to {stats_path}")
        return True


class CheckpointWithVecNormalizeCallback(BaseCallback):
    """
    Callback that saves model checkpoints and VecNormalize stats periodically.
    """
    def __init__(self, save_freq: int, save_path: str, vecnormalize_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vecnormalize_env = vecnormalize_env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            os.makedirs(self.save_path, exist_ok=True)
            
            # Save PPO model
            model_file = os.path.join(self.save_path, f"model_step_{step}.zip")
            self.model.save(model_file)
            
            # Save VecNormalize stats
            stats_file = os.path.join(self.save_path, f"vecnormalize_step_{step}.pkl")
            self.vecnormalize_env.save(stats_file)
            
            logger.info(f"[Callback] Checkpoint saved at step {step}: model={model_file}, stats={stats_file}")
        return True


def run_training(config_path: str = "rl/config/rl_config.yaml"):
    """
    Runs the PPO training pipeline using real map data.
    """
    # 1. Load config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Check Real Data Policy
    map_path = config.get("data", {}).get("geo_network_map", "data/processed/geo_network_map.parquet")
    metadata_path = config.get("data", {}).get("metadata", "data/processed/geo_network_metadata.json")

    if not os.path.exists(map_path) or not os.path.exists(metadata_path):
        print("\nERROR: Real Geo-Network Map not found.")
        print(f"Please build {map_path} before training.\n")
        logger.error("Training stopped: missing real Geo-Network Map files.")
        sys.exit(1)

    # 3. Setup reproducibility seed
    seed = config.get("training", {}).get("seed", 42)
    set_global_seed(seed)

    # 3b. Device selection (CUDA for parallel PPO processing if available)
    device_cfg = config.get("training", {}).get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg
    logger.info(f"Using compute device: {device}")
    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 4. Load & Validate Data
    logger.info("Loading and validating Geo-Network Map...")
    df, metadata = load_geo_network_map(map_path, metadata_path)
    summary = get_data_quality_summary(df)
    print_quality_summary_report(summary)

    # Verify grid cell counts
    if summary["total_cells"] == 0:
        logger.error("Geo-Network Map contains no cells.")
        sys.exit(1)

    # 5. Fit & Save Feature Processor
    best_model_dir = "rl/models/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    
    fp_path = os.path.join(best_model_dir, "feature_processor.joblib")
    feature_processor = FeatureProcessor()
    feature_processor.fit(df, metadata)
    feature_processor.save(fp_path)

    # 6. Generate/Load fixed evaluation validation missions
    generate_and_save_scenarios(df, metadata, num_scenarios=20)
    val_missions = load_validation_scenarios()

    # 7. Create environments
    logger.info("Creating Training Environment...")
    train_raw_env = DroneNetworkEnv(
        grid_df=df,
        metadata=metadata,
        config=config,
        feature_processor=feature_processor
    )
    
    # Gymnasium check_env before PPO wrap
    logger.info("Running Gymnasium environment compliance check...")
    check_env(train_raw_env)
    logger.info("Gymnasium compliance check passed.")

    # Wrap training environment
    train_mon_env = Monitor(train_raw_env)
    train_vec_env = DummyVecEnv([lambda: train_mon_env])
    train_norm_env = VecNormalize(train_vec_env, norm_obs=True, norm_obs_keys=None, norm_reward=False, clip_obs=10.0)

    # Create Evaluation Environment
    logger.info("Creating Evaluation Environment...")
    eval_raw_env = DroneNetworkEnv(
        grid_df=df,
        metadata=metadata,
        config=config,
        feature_processor=feature_processor
    )
    # Enable evaluation mode with fixed validation missions
    eval_raw_env.set_eval_mode(val_missions)
    
    eval_mon_env = Monitor(eval_raw_env)
    eval_vec_env = DummyVecEnv([lambda: eval_mon_env])
    
    # Normalize evaluation observations using same stats as training
    eval_norm_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    # Synchronize stats
    eval_norm_env.obs_rms = train_norm_env.obs_rms

    # 8. Setup PPO Hyperparameters
    ppo_config = config.get("ppo", {})
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "env": train_norm_env,
        "learning_rate": float(ppo_config.get("learning_rate", 0.0003)),
        "n_steps": int(ppo_config.get("n_steps", 2048)),
        "batch_size": int(ppo_config.get("batch_size", 64)),
        "n_epochs": int(ppo_config.get("n_epochs", 10)),
        "gamma": float(ppo_config.get("gamma", 0.99)),
        "gae_lambda": float(ppo_config.get("gae_lambda", 0.95)),
        "clip_range": float(ppo_config.get("clip_range", 0.2)),
        "ent_coef": float(ppo_config.get("ent_coef", 0.01)),
        "vf_coef": float(ppo_config.get("vf_coef", 0.5)),
        "max_grad_norm": float(ppo_config.get("max_grad_norm", 0.5)),
        "seed": seed,
        "device": device,
        "tensorboard_log": "rl/logs/",
        "verbose": 1
    }
    
    model = PPO(**ppo_kwargs)
    logger.info("PPO model initialized.")

    # 9. Setup Callbacks
    callbacks_list = []

    # Custom Metrics CSV/TB Callback
    metrics_csv = "rl/logs/training_metrics.csv"
    callbacks_list.append(CustomMetricsCallback(csv_log_path=metrics_csv))

    # Periodic checkpoint callback (runs every X steps)
    train_config = config.get("training", {})
    checkpoint_freq = int(train_config.get("checkpoint_frequency", 50000))
    checkpoint_dir = "rl/models/checkpoints"
    checkpoint_cb = CheckpointWithVecNormalizeCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        vecnormalize_env=train_norm_env
    )
    callbacks_list.append(checkpoint_cb)

    # Eval callback on fixed validation set
    eval_freq = int(train_config.get("evaluation_frequency", 25000))
    save_vec_best_cb = SaveVecNormalizeOnBestCallback(
        vecnormalize_env=train_norm_env,
        save_dir=best_model_dir
    )
    eval_cb = EvalCallback(
        eval_env=eval_norm_env,
        callback_on_new_best=save_vec_best_cb,
        n_eval_episodes=len(val_missions),
        eval_freq=eval_freq,
        best_model_save_path=best_model_dir,
        log_path="rl/logs/",
        deterministic=True
    )
    callbacks_list.append(eval_cb)

    # Curriculum callback (optional)
    curr_config = config.get("curriculum", {})
    total_timesteps = int(train_config.get("total_timesteps", 500000))
    
    if curr_config.get("enabled", True):
        stages = curr_config.get("stages", [])
        if stages:
            curr_cb = CurriculumCallback(
                train_env=train_norm_env,
                stages=stages,
                total_timesteps=total_timesteps
            )
            callbacks_list.append(curr_cb)

    # Combine all callbacks
    all_callbacks = CallbackList(callbacks_list)

    # 10. Run PPO Training
    logger.info(f"Starting PPO training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            tb_log_name="PPO_drone"
        )
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model state...")
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        raise e

    # 11. Save Final Model & Stats
    final_model_path = os.path.join(best_model_dir, "final_model.zip")
    model.save(final_model_path)
    train_norm_env.save(os.path.join(best_model_dir, "final_model_vecnormalize.pkl"))
    logger.info(f"Saved final model and statistics to {best_model_dir}")

if __name__ == "__main__":
    run_training()
