# config/paths.py
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

# Data Paths
RAW_DATA_DIR = get_project_root() / "data" / "raw"
PROCESSED_DATA_DIR = get_project_root() / "data" / "processed"
SIMULATIONS_DIR = get_project_root() / "data" / "simulations"

# Model Paths
MODELS_DIR = get_project_root() / "models"
RL_CHECKPOINTS = MODELS_DIR / "reinforcement_learning"

# Logging
LOGS_DIR = get_project_root() / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SIMULATIONS_DIR, MODELS_DIR, RL_CHECKPOINTS, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
