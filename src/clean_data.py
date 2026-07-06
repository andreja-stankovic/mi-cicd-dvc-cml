from pathlib import Path

import pandas as pd
import yaml


RAW_DATA_PATH = Path("data/raw/hotel_bookings.csv")
CLEANED_DATA_PATH = Path("data/processed/cleaned_hotel_bookings.csv")
PARAMS_PATH = Path("params.yaml")


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clean_data() -> None:
    params = load_params()
    cleaning_params = params["cleaning"]

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    if "is_canceled" not in df.columns:
        raise ValueError("Target column 'is_canceled' is missing from the dataset.")

    initial_rows = len(df)

    # Remove fully duplicated rows
    df = df.drop_duplicates()

    # Fill missing values
    df["children"] = df["children"].fillna(cleaning_params["fill_children_with"])
    df["country"] = df["country"].fillna(cleaning_params["fill_country_with"])
    df["agent"] = df["agent"].fillna(cleaning_params["fill_agent_with"])
    df["company"] = df["company"].fillna(cleaning_params["fill_company_with"])

    # Remove reservations with no guests
    if cleaning_params["remove_zero_guests"]:
        total_guests = df["adults"] + df["children"] + df["babies"]
        df = df[total_guests > 0]

    # Filter invalid or extreme ADR values
    adr_min = cleaning_params["adr_min"]
    adr_max = cleaning_params["adr_max"]
    df = df[(df["adr"] >= adr_min) & (df["adr"] <= adr_max)]

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print("Cleaning completed.")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Removed rows: {initial_rows - len(df)}")
    print(f"Output saved to: {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    clean_data()