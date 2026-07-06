import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/model.pkl")
REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TARGET_COLUMN = "is_canceled"


def evaluate_model() -> None:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {TEST_DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(y_test, y_pred)
    CLASSIFICATION_REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not canceled", "Canceled"],
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print("Evaluation completed.")
    print(json.dumps(metrics, indent=4))
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate_model()