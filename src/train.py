from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


TRAIN_DATA_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")
PARAMS_PATH = Path("params.yaml")
TARGET_COLUMN = "is_canceled"


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model() -> None:
    params = load_params()
    model_params = params["model"]
    split_params = params["split"]

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {TRAIN_DATA_PATH}")

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    if model_params["type"] != "random_forest":
        raise ValueError("Only 'random_forest' model type is supported in this project.")

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=split_params["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Training rows: {X_train.shape[0]}")


if __name__ == "__main__":
    train_model()