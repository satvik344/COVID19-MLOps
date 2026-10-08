"""Lab 5 input-data validation with a Pandera check when available and a pandas fallback."""
import os
import pandas as pd

EXPECTED_COLUMNS = [
    "Date", "Country/Region", "Confirmed", "Deaths", "Recovered", "Active",
    "New cases", "New deaths", "New recovered", "WHO Region"
]

def validate_data(path="data/full_grouped.csv"):
    df = pd.read_csv(path)
    errors = []
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing: errors.append(f"Missing columns: {missing}")
    if extra: errors.append(f"Unexpected columns: {extra}")

    # Cumulative counts should never be negative.
    for col in ["Confirmed", "Deaths", "Recovered"]:
        if col in df:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any():
                errors.append(f"{col} contains non-numeric values.")
            elif (series < 0).any():
                errors.append(f"{col} contains negative cumulative counts.")

    # Daily/corrected fields can legitimately be negative in this source because
    # historical corrections may subtract previously reported values.
    for col in ["Active", "New cases", "New deaths", "New recovered"]:
        if col in df:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any():
                errors.append(f"{col} contains non-numeric values.")

    if "Date" in df and pd.to_datetime(df["Date"], errors="coerce").isna().any():
        errors.append("Date contains invalid values.")

    if "Country/Region" in df:
        if "India" not in set(df["Country/Region"].dropna()):
            errors.append("India is not present in the dataset.")

    os.makedirs("artifacts", exist_ok=True)
    report = {"rows": len(df), "columns": len(df.columns), "status": "PASSED" if not errors else "FAILED", "errors": errors}
    pd.DataFrame({"error": errors}).to_csv("artifacts/input_validation_errors.csv", index=False)
    if errors:
        raise ValueError("Input validation failed: " + " | ".join(errors))
    print("[SUCCESS] Input data validation PASSED.")
    return True

if __name__ == "__main__":
    validate_data()
