"""Run all three MLOps experiments in order."""
import subprocess
import sys

for script in [
    "pipelines/run_lab3_baseline.py",
    "pipelines/run_lab4_tracking.py",
    "pipelines/run_lab5_pipeline.py",
]:
    print(f"\n>>> {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(f"Pipeline failed: {script}")
print("\n[SUCCESS] All experiments completed.")
