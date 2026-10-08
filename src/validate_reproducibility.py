"""Lab 4 reproducibility check."""
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_once(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred) ** 0.5

def main():
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    rmse1 = train_once(X_train, y_train, X_test, y_test)
    rmse2 = train_once(X_train, y_train, X_test, y_test)

    os.makedirs("artifacts", exist_ok=True)
    same = bool(np.isclose(rmse1, rmse2, rtol=0, atol=1e-12))
    with open("artifacts/reproducibility_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Execution 1 RMSE: {rmse1:.12f}\n")
        f.write(f"Execution 2 RMSE: {rmse2:.12f}\n")
        f.write(f"Reproducible: {same}\n")
    if not same:
        raise RuntimeError("Reproducibility validation failed.")
    print(f"[SUCCESS] Reproducible. RMSE={rmse1:.6f}")

if __name__ == "__main__":
    main()
