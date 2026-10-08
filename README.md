# COVID-19 India ML + MLOps Project

This is the main COVID-19 machine-learning project, upgraded with the same **Lab 3, Lab 4 and Lab 5 experiment workflow** implemented in the Churn project.

## Project objective

Predict India's current cumulative COVID-19 confirmed cases from the previous day's confirmed cases.

The original beginner ML workflow is retained:
- Data loading and cleaning
- India selection
- Feature engineering
- EDA
- Chronological train/test split
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- MAE, RMSE and R² comparison
- Sample prediction

## MLOps experiments added

### Lab 3 — Baseline ML Pipeline
`pipelines/run_lab3_baseline.py`

Runs:
1. `src/preprocess.py`
2. `src/train.py`
3. `src/evaluate.py`

Outputs:
- Processed train/test arrays
- Dataset metadata
- Three trained model files
- Model comparison CSV
- Test predictions
- Best-model report
- Sample prediction

### Lab 4 — MLflow Experiment Tracking
`pipelines/run_lab4_tracking.py`

Runs:
1. Baseline preprocessing
2. `src/train_mlflow.py`
3. `src/validate_reproducibility.py`

Tracks each regression model in MLflow:
- Parameters
- MAE
- RMSE
- R²
- Actual-vs-predicted diagnostic plots
- Serialized sklearn model artifacts

MLflow tracking data is stored locally in the SQLite database `mlflow.db`.

### Lab 5 — Production Data Pipeline
`pipelines/run_lab5_pipeline.py`

Runs:
1. `src/validate_data.py`
2. `src/preprocess_pipeline.py`
3. `src/validate_outputs.py`

Adds:
- Input schema/data-quality validation
- Production-style scikit-learn preprocessing pipeline
- Imputation
- Standardization
- Saved preprocessing artifact
- Metadata
- Output validation
- Shape and NaN checks

## Project structure

```text
COVID19_MLOps_Final/
│
├── data/
│   ├── full_grouped.csv
│   └── processed/
│       ├── dataset_metadata.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── notebooks/
│   └── project_implementation.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_mlflow.py
│   ├── validate_reproducibility.py
│   ├── validate_data.py
│   ├── preprocess_pipeline.py
│   └── validate_outputs.py
│
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   └── run_lab5_pipeline.py
│
├── models/
├── outputs/
├── logs/
├── artifacts/
│
├── beginner_project.py
├── run_all.py
├── requirements.txt
├── README.md
├── VIVA_CHEAT_SHEET.md
└── Research_Paper_COVID19_ML.pdf
```

## Setup in VS Code

Open the unzipped folder in VS Code.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the experiments

Run Lab 3:

```powershell
python .\pipelines\run_lab3_baseline.py
```

Run Lab 4:

```powershell
python .\pipelines\run_lab4_tracking.py
```

Run Lab 5:

```powershell
python .\pipelines\run_lab5_pipeline.py
```

Or run everything:

```powershell
python .\run_all.py
```

The original entry point also runs Lab 3:

```powershell
python .\beginner_project.py
```

## View MLflow

After Lab 4:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Then open the local MLflow address shown in the terminal.

## Dataset validation note

The supplied COVID dataset contains some negative values in correction-sensitive fields such as `New deaths`, `New recovered`, and `Active`. Lab 5 allows these source corrections but requires cumulative `Confirmed`, `Deaths`, and `Recovered` counts to be non-negative.

## Important note about the time-series split

The project uses `shuffle=False` so earlier dates are used for training and later dates are used for testing. This avoids randomly mixing future observations into the training set.

## Main files to explain in viva

- `beginner_project.py` — original beginner entry point
- `src/preprocess.py` — baseline preprocessing
- `src/train.py` — baseline model training
- `src/evaluate.py` — metrics and comparison
- `src/train_mlflow.py` — experiment tracking
- `src/validate_reproducibility.py` — repeatability check
- `src/validate_data.py` — input validation
- `src/preprocess_pipeline.py` — production preprocessing
- `src/validate_outputs.py` — output validation
- `pipelines/` — reproducible experiment runners
