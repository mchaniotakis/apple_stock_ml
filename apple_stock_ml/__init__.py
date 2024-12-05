"""Apple stock machine learning package."""

# set the seed for reproducable results
import os
import torch
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__version__ = "0.1.0"
