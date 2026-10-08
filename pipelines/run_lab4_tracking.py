import subprocess, sys

def execute_pipeline():
    try:
        import mlflow  # noqa: F401
    except ImportError:
        raise SystemExit(
            "MLflow is not installed. Run: pip install -r requirements.txt"
        )
    print("=== Lab 4: MLflow Experiment Tracking ===")
    for script in ["src/preprocess.py", "src/train_mlflow.py", "src/validate_reproducibility.py"]:
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise SystemExit(f"Lab 4 failed at {script}")
    print("[SUCCESS] Lab 4 completed.")

if __name__ == "__main__":
    execute_pipeline()
