import os
from stable_baselines3.common.vec_env import VecNormalize
from rl.utils.logger import setup_logger

logger = setup_logger("checkpointing")

def save_checkpoint(model, env, checkpoint_dir: str, step: int):
    """
    Saves model checkpoint and VecNormalize environment statistics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoint_dir, f"model_step_{step}.zip")
    model.save(model_path)
    logger.info(f"Saved model checkpoint to {model_path}")
    
    # Save VecNormalize stats if present
    # Check if env is VecNormalize or nested VecNormalize
    unwrapped = env
    while hasattr(unwrapped, "venv"):
        if isinstance(unwrapped, VecNormalize):
            break
        unwrapped = unwrapped.venv
        
    if isinstance(unwrapped, VecNormalize):
        vec_path = os.path.join(checkpoint_dir, f"vecnormalize_step_{step}.pkl")
        unwrapped.save(vec_path)
        logger.info(f"Saved VecNormalize statistics to {vec_path}")

def save_best_model(model, env, best_model_dir: str):
    """
    Saves the best model and VecNormalize statistics.
    """
    os.makedirs(best_model_dir, exist_ok=True)
    
    model_path = os.path.join(best_model_dir, "best_model.zip")
    model.save(model_path)
    logger.info(f"Saved best model to {model_path}")
    
    unwrapped = env
    while hasattr(unwrapped, "venv"):
        if isinstance(unwrapped, VecNormalize):
            break
        unwrapped = unwrapped.venv
        
    if isinstance(unwrapped, VecNormalize):
        vec_path = os.path.join(best_model_dir, "best_model_vecnormalize.pkl")
        unwrapped.save(vec_path)
        logger.info(f"Saved best VecNormalize statistics to {vec_path}")
