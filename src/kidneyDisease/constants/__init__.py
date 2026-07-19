from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

MLFLOW_URI = os.getenv("MLFLOW_URI")
MLFLOW_EXPERIMENT_NAME = "Kidney Disease Classification"