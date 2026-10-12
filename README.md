# Machine Learning Algorithms

A collection of hands-on machine-learning notebooks covering data preprocessing, exploratory analysis, linear regression from scratch, gradient descent, feature scaling, and introductory classification.

The notebooks are educational experiments rather than a packaged Python library. They show the mathematics behind common models and compare implementations written with NumPy against scikit-learn.

## Contents

| Notebook | Topic |
| --- | --- |
| `advertising_linear_regression_gradient_descent.ipynb` | Univariate advertising regression with a custom gradient-descent implementation |
| `advertising_regression_from_scratch.ipynb` | Three separate advertising regressions: TV, Radio, and Newspaper versus Sales |
| `data_preprocessing_workflow.ipynb` | Missing values, categorical encoding, train/test splitting, and feature scaling |
| `iris_regression_gradient_descent.ipynb` | Predicting Iris petal width from petal length with gradient descent |
| `multiple_linear_regression_from_scratch.ipynb` | Multiple linear regression using NumPy and scikit-learn |
| `tp3_multiple_linear_regression.ipynb` | Advertising-sales regression, visualisation, and model comparison |
| `tp4_multiple_linear_regression_scaling.ipynb` | Standardised multiple regression and convergence analysis |
| `iris_classification_and_torch_regression.ipynb` | Exploratory Iris classification plus an experimental PyTorch regression section |

## Data

The `data/` directory contains the datasets used by the notebooks:

- `Advertising.csv` — TV, Radio, and Newspaper advertising budgets with Sales.
- `Data.csv` — a small preprocessing example with Country, Age, Salary, and Purchased.
- `iris.data` / `bezdekIris.data` — the classic Iris dataset in UCI format.
- `diabetes.csv`, `titanic.csv`, `tips.csv`, and `US-pumpkins.csv` — additional practice datasets.
- `iris.names` and `Index` — metadata distributed with the Iris dataset.

Some notebooks were originally created in Google Colab or Kaggle and retain absolute paths such as `/content/...` or `../input/...`. Update those paths to `data/<filename>` before executing them locally. The notebooks are intentionally preserved as coursework artifacts; see [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md) for the remaining technical notes.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch Jupyter from the repository root so relative paths resolve consistently:

```bash
jupyter lab
```

## Reproducibility notes

- Most train/test splits use `random_state=42`.
- The notebooks use both custom gradient descent and scikit-learn reference models.
- Notebook outputs are retained where they document the original work; rerun notebooks after correcting their data paths.
- The PyTorch notebook assumes a CUDA device in its experimental section and may require changing the device to `cpu`.

## Repository status

This repository is prepared as a learning portfolio. Before publishing, review notebook outputs for accidental personal information, rerun the selected notebooks from a clean environment, and commit only the notebooks and datasets that you intend to share.
