"""Lab 5 validation of transformed production outputs."""
import json
import os
import numpy as np

def validate_preprocessing_outputs():
    paths = {
        "X_train": "data/processed/X_train_pipeline.npy",
        "X_test": "data/processed/X_test_pipeline.npy",
        "y_train": "data/processed/y_train_pipeline.npy",
        "y_test": "data/processed/y_test_pipeline.npy",
    }
    arrays = {k: np.load(v) for k, v in paths.items()}
    errors = []
    if np.isnan(arrays["X_train"]).any() or np.isnan(arrays["X_test"]).any():
        errors.append("NaNs detected after imputation.")
    if arrays["X_train"].shape[1] != arrays["X_test"].shape[1]:
        errors.append("Train/test feature dimensions differ.")
    if len(arrays["X_train"]) != len(arrays["y_train"]):
        errors.append("X_train/y_train row mismatch.")
    if len(arrays["X_test"]) != len(arrays["y_test"]):
        errors.append("X_test/y_test row mismatch.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "errors": errors,
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/production_output_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    if errors:
        raise RuntimeError("Production output validation failed: " + "; ".join(errors))
    print("[SUCCESS] Production output validation PASSED.")

if __name__ == "__main__":
    validate_preprocessing_outputs()
