import random
import numpy as np
import torch
from rl.utils.logger import setup_logger

logger = setup_logger("seed")

def set_global_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, PyTorch, and GPU operations.
    Ensures reproducibility of PPO training and evaluation steps.
    """
    logger.info(f"Setting global seed to: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    logger.info("Global seeds configured successfully.")
