"""
Lab 3 baseline preprocessing for the COVID-19 India regression project.
Creates a chronological train/test split and saves reusable processed arrays.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/full_grouped.csv"

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    required = ["Date", "Country/Region", "Confirmed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["Country/Region"].eq("India")].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Confirmed"])
    df = df.sort_values("Date").drop_duplicates()
    df["Confirmed"] = pd.to_numeric(df["Confirmed"], errors="coerce").fillna(0)
    df["Previous_Day_Confirmed"] = df["Confirmed"].shift(1)
    df = df.dropna(subset=["Previous_Day_Confirmed"]).reset_index(drop=True)
    return df

def run_preprocessing():
    print("[INFO] Lab 3: baseline preprocessing")
    df = load_and_prepare()
    X = df[["Previous_Day_Confirmed"]]
    y = df["Confirmed"]

    # Chronological split: earlier observations train the model; later observations test it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train.to_numpy(dtype=float))
    np.save("data/processed/X_test.npy", X_test.to_numpy(dtype=float))
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=float))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=float))

    metadata = {
        "dataset_name": "COVID-19 Global Time Series",
        "country": "India",
        "target": "Confirmed",
        "feature": "Previous_Day_Confirmed",
        "split": {"test_size": 0.20, "shuffle": False},
        "rows_after_feature_engineering": int(len(df)),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "date_range": [df["Date"].min().strftime("%Y-%m-%d"), df["Date"].max().strftime("%Y-%m-%d")]
    }
    with open("data/processed/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Train: {X_train.shape}; Test: {X_test.shape}")

if __name__ == "__main__":
    run_preprocessing()
