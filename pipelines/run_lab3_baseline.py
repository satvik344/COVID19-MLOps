import subprocess, sys

def execute_pipeline():
    print("=== Lab 3: Baseline ML Pipeline ===")
    for script in ["src/preprocess.py", "src/train.py", "src/evaluate.py"]:
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise SystemExit(f"Lab 3 failed at {script}")
    print("[SUCCESS] Lab 3 completed.")

if __name__ == "__main__":
    execute_pipeline()
