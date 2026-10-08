"""Lab 3 baseline training: three regression models."""
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_models():
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    os.makedirs("models", exist_ok=True)

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        path = f"models/{name}.pkl"
        joblib.dump(model, path)
        print(f"[SUCCESS] Saved {path}")
    return models

if __name__ == "__main__":
    train_models()
