# Project Review

## Scope

The repository was reviewed as a collection of Jupyter notebooks and small practice datasets. The review covered notebook metadata, cell structure, imports, data references, output presence, filenames, and the working-tree layout.

## Findings

### Strengths

- The work demonstrates the complete learning workflow: exploration, preprocessing, train/test splitting, model training, visualisation, and evaluation.
- Several notebooks implement linear regression and gradient descent from first principles, then compare the result with scikit-learn.
- Fixed random seeds and explicit test splits are used in most modelling exercises.
- The advertising dataset is reused consistently across the regression exercises, making model comparisons easy to follow.

### Items to address before running locally

1. Dataset paths are not yet fully standardised. Original Colab/Kaggle paths such as `/content/Advertising.csv` and `../input/preprocessingdataset/Data.csv` should be changed to `data/Advertising.csv` and `data/Data.csv`.
2. `iris_regression_gradient_descent.ipynb` expects `iris.csv`, while the repository provides the UCI-formatted `data/iris.data`. Load it with explicit column names before running the notebook.
3. The PyTorch section of `iris_classification_and_torch_regression.ipynb` selects `cuda` unconditionally. Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` for portability.
4. A few notebooks contain installation commands such as `%pip install` and `!pip install`. Dependencies should be installed from `requirements.txt`, not during notebook execution.
5. Some notebooks are exploratory duplicates. Keep the clearest one or retain the others as historical coursework with a short note explaining the distinction.
6. Notebook outputs are large in places. Before a final portfolio release, clear and rerun outputs selectively to reduce repository size and remove stale results.

## Recommended publication order

1. `data_preprocessing_workflow.ipynb`
2. `advertising_linear_regression_gradient_descent.ipynb`
3. `tp4_multiple_linear_regression_scaling.ipynb`
4. `iris_classification_and_torch_regression.ipynb` (optional/experimental)

## Validation performed

- All notebooks were parsed as valid JSON.
- Notebook cell counts, imports, data references, and existing outputs were inventoried.
- Dataset headers and the repository layout were checked.
- No Git repository or remote was present at the time of review, so no push was performed.
