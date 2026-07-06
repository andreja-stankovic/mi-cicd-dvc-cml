from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


CLEANED_DATA_PATH = Path("data/processed/cleaned_hotel_bookings.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
PARAMS_PATH = Path("params.yaml")


NUMERICAL_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

TARGET_COLUMN = "is_canceled"


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def transform_data() -> None:
    params = load_params()
    split_params = params["split"]

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CLEANED_DATA_PATH}")

    df = pd.read_csv(CLEANED_DATA_PATH)

    required_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Convert categorical variables into numeric dummy variables
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    transformed_df = X.copy()
    transformed_df[TARGET_COLUMN] = y.values

    train_df, test_df = train_test_split(
        transformed_df,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=transformed_df[TARGET_COLUMN],
    )

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("Transformation completed.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train data saved to: {TRAIN_DATA_PATH}")
    print(f"Test data saved to: {TEST_DATA_PATH}")


if __name__ == "__main__":
    transform_data()