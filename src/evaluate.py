"""Lab 3 evaluation and error analysis."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_NAMES = ["linear_regression", "decision_tree", "random_forest"]

def evaluate_models():
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    rows = []
    predictions = {"Actual": y_test}

    for name in MODEL_NAMES:
        model = joblib.load(f"models/{name}.pkl")
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        predictions[name] = pred

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv("outputs/test_predictions.csv", index=False)

    best = results.iloc[0]
    with open("outputs/best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model by RMSE: {best['Model']}\n")
        f.write(f"RMSE: {best['RMSE']:.4f}\n")
        f.write(f"MAE: {best['MAE']:.4f}\n")
        f.write(f"R2: {best['R2']:.6f}\n")

    print("\n[INFO] Lab 3 Model Comparison")
    print(results.to_string(index=False))
    print(f"\n[SUCCESS] Best model by RMSE: {best['Model']}")

    # Sample unseen prediction using the selected model.
    sample = pd.DataFrame({"Previous_Day_Confirmed": [1_000_000]})
    selected = joblib.load(f"models/{best['Model']}.pkl")
    sample_prediction = float(selected.predict(sample.to_numpy())[0])
    with open("outputs/sample_prediction.txt", "w", encoding="utf-8") as f:
        f.write("Previous day confirmed cases: 1000000\n")
        f.write(f"Predicted confirmed cases: {sample_prediction:.0f}\n")
    return results

if __name__ == "__main__":
    evaluate_models()
