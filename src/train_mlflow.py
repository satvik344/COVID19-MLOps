"""Lab 4: MLflow experiment tracking for the three regression models."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MLFLOW_DB = "sqlite:///mlflow.db"

def train_and_track():
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment("COVID19_India_Regression")

    model_specs = {
        "LinearRegression": (LinearRegression(), {}),
        "DecisionTree": (DecisionTreeRegressor(max_depth=5, random_state=42), {"max_depth": 5, "random_state": 42}),
        "RandomForest": (RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
                         {"n_estimators": 100, "max_depth": 8, "random_state": 42}),
    }

    os.makedirs("artifacts", exist_ok=True)
    summary = []
    for run_name, (model, params) in model_specs.items():
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_family", run_name)
            mlflow.log_params(params)
            mlflow.log_param("feature", "Previous_Day_Confirmed")
            mlflow.log_param("target", "Confirmed")
            mlflow.log_param("split_strategy", "chronological_80_20")
            mlflow.log_dict({"project": "COVID19 India regression"}, "project_metadata.json")

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = {
                "mae": mean_absolute_error(y_test, pred),
                "rmse": mean_squared_error(y_test, pred) ** 0.5,
                "r2": r2_score(y_test, pred),
            }
            mlflow.log_metrics(metrics)

            # Diagnostic prediction plot.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(y_test, label="Actual")
            ax.plot(pred, label="Predicted")
            ax.set_title(f"Actual vs Predicted - {run_name}")
            ax.set_xlabel("Test observation")
            ax.set_ylabel("Confirmed cases")
            ax.legend()
            plot_path = f"artifacts/{run_name.lower()}_actual_vs_predicted.png"
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            mlflow.log_artifact(plot_path, artifact_path="plots")
            mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
)

            summary.append({"Model": run_name, **metrics})
            print(f"[SUCCESS] MLflow run {run_name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    pd.DataFrame(summary).sort_values("rmse").to_csv("outputs/mlflow_summary.csv", index=False)
    print("[SUCCESS] MLflow tracking data saved in mlflow.db")
    print("[SUCCESS] Lab 4 completed.")

if __name__ == "__main__":
    train_and_track()
