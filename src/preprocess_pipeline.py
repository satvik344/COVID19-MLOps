"""Lab 5 production preprocessing using a scikit-learn Pipeline."""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_and_run_pipeline():
    df = pd.read_csv("data/full_grouped.csv")
    df = df[df["Country/Region"].eq("India")].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Confirmed"]).sort_values("Date").drop_duplicates()
    df["Confirmed"] = pd.to_numeric(df["Confirmed"], errors="coerce")
    df["Previous_Day_Confirmed"] = df["Confirmed"].shift(1)
    df = df.dropna(subset=["Previous_Day_Confirmed"])

    X = df[["Previous_Day_Confirmed"]]
    y = df["Confirmed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    feature_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_final = feature_pipeline.fit_transform(X_train)
    X_test_final = feature_pipeline.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_pipeline.npy", X_train_final)
    np.save("data/processed/X_test_pipeline.npy", X_test_final)
    np.save("data/processed/y_train_pipeline.npy", y_train.to_numpy(dtype=float))
    np.save("data/processed/y_test_pipeline.npy", y_test.to_numpy(dtype=float))
    joblib.dump(feature_pipeline, "models/production_preprocessor.pkl")

    metadata = {
        "pipeline": ["SimpleImputer(median)", "StandardScaler"],
        "feature": ["Previous_Day_Confirmed"],
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
    }
    with open("artifacts/production_pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("[SUCCESS] Production sklearn preprocessing pipeline completed.")

if __name__ == "__main__":
    build_and_run_pipeline()
