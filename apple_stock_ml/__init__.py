"""Apple stock machine learning package."""

# set the seed for reproducable results
import os
import torch
import random
import numpy as np

SEED = 42


# Add this function after the imports and before any other code
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seeds(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__version__ = "0.1.0"
