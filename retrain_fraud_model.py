import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection.fraud_model import build_and_save_model

if __name__ == "__main__":
    p = build_and_save_model(model_type="random_forest")
    print("SAVED", p)
