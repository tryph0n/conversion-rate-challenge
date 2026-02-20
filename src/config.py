"""Project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = INPUT_DIR / "conversion_data_train.csv"
TEST_FILE = INPUT_DIR / "conversion_data_test.csv"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
METADATA_FILE = MODELS_DIR / "metadata.joblib"

TARGET = "converted"
RANDOM_STATE = 42
