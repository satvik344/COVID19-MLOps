import subprocess, sys

def execute_pipeline():
    print("=== Lab 5: Production Data Pipeline ===")
    for script in ["src/validate_data.py", "src/preprocess_pipeline.py", "src/validate_outputs.py"]:
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise SystemExit(f"Lab 5 failed at {script}")
    print("[SUCCESS] Lab 5 completed.")

if __name__ == "__main__":
    execute_pipeline()
