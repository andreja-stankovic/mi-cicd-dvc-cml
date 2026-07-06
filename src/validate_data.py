import json
from pathlib import Path

import pandas as pd
import yaml


CLEANED_DATA_PATH = Path("data/processed/cleaned_hotel_bookings.csv")
PARAMS_PATH = Path("params.yaml")
REPORTS_DIR = Path("reports")
VALIDATION_REPORT_PATH = REPORTS_DIR / "data_validation.json"

TARGET_COLUMN = "is_canceled"

KEY_COLUMNS = [
    "is_canceled",
    "lead_time",
    "adults",
    "children",
    "babies",
    "country",
    "adr",
]


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def add_check(checks: list, name: str, passed: bool, details: str) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "details": details,
        }
    )


def validate_data() -> None:
    params = load_params()
    cleaning_params = params["cleaning"]

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CLEANED_DATA_PATH}")

    df = pd.read_csv(CLEANED_DATA_PATH)

    checks = []

    # Check 1: dataset is not empty
    add_check(
        checks,
        "Dataset is not empty",
        len(df) > 0,
        f"Row count: {len(df)}",
    )

    # Check 2: target column exists
    target_exists = TARGET_COLUMN in df.columns
    add_check(
        checks,
        "Target column exists",
        target_exists,
        f"Target column: {TARGET_COLUMN}",
    )

    # Check 3: required key columns exist
    missing_key_columns = [column for column in KEY_COLUMNS if column not in df.columns]
    add_check(
        checks,
        "Key columns exist",
        len(missing_key_columns) == 0,
        f"Missing key columns: {missing_key_columns}",
    )

    if missing_key_columns:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        validation_report = {
            "status": "failed",
            "row_count": int(len(df)),
            "checks": checks,
        }

        with VALIDATION_REPORT_PATH.open("w", encoding="utf-8") as file:
            json.dump(validation_report, file, indent=4)

        raise ValueError("Data validation failed because some key columns are missing.")

    # Check 4: no missing values in key columns
    missing_values = df[KEY_COLUMNS].isna().sum()
    missing_values_dict = {
        column: int(value)
        for column, value in missing_values.items()
        if int(value) > 0
    }

    add_check(
        checks,
        "No missing values in key columns",
        len(missing_values_dict) == 0,
        f"Missing values: {missing_values_dict}",
    )

    # Check 5: no duplicate rows
    duplicate_count = int(df.duplicated().sum())
    add_check(
        checks,
        "No duplicate rows",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}",
    )

    # Check 6: every reservation has at least one guest
    total_guests = df["adults"] + df["children"] + df["babies"]
    zero_guest_rows = int((total_guests <= 0).sum())

    add_check(
        checks,
        "Every reservation has at least one guest",
        zero_guest_rows == 0,
        f"Rows with zero guests: {zero_guest_rows}",
    )

    # Check 7: ADR is in expected range
    adr_min = cleaning_params["adr_min"]
    adr_max = cleaning_params["adr_max"]

    invalid_adr_rows = int(
        ((df["adr"] < adr_min) | (df["adr"] > adr_max)).sum()
    )

    add_check(
        checks,
        "ADR values are in expected range",
        invalid_adr_rows == 0,
        f"Invalid ADR rows: {invalid_adr_rows}, expected range: [{adr_min}, {adr_max}]",
    )

    # Check 8: target column has only binary values
    target_values = sorted(df[TARGET_COLUMN].dropna().unique().tolist())

    add_check(
        checks,
        "Target column contains only binary values",
        set(target_values).issubset({0, 1}),
        f"Observed target values: {target_values}",
    )

    # Check 9: both classes are present
    class_distribution = {
        str(class_value): int(count)
        for class_value, count in df[TARGET_COLUMN].value_counts().sort_index().items()
    }

    add_check(
        checks,
        "Both target classes are present",
        len(class_distribution) == 2,
        f"Class distribution: {class_distribution}",
    )

    all_checks_passed = all(check["passed"] for check in checks)

    validation_report = {
        "status": "passed" if all_checks_passed else "failed",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "class_distribution": class_distribution,
        "checks": checks,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with VALIDATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(validation_report, file, indent=4)

    print("Data validation completed.")
    print(json.dumps(validation_report, indent=4))

    if not all_checks_passed:
        failed_checks = [
            check["name"]
            for check in checks
            if not check["passed"]
        ]

        raise ValueError(f"Data validation failed. Failed checks: {failed_checks}")


if __name__ == "__main__":
    validate_data()