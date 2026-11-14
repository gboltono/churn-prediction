# Customer Churn Prediction

MAT 311 Intro to Data Science final project (Fall 2025): an end-to-end Python pipeline predicting churn for 300K+ customers in a Kaggle competition.

## Results

On a 30% validation set (90,938 customers):

| Model | ROC AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| **KNN (k = 10, scaled)** | **0.921** | **0.757** | 0.742 | 0.907 |
| Decision tree | 0.821 | 0.750 | 0.746 | 0.896 |

## Pipeline

1. **Clean** (`src/data/preprocessing.py`): drop leaking `Customer Status`, impute missing values, derive `Payment Delay` from dates.
2. **Features** (`src/features/build_features.py`): one-hot encode categories.
3. **EDA** (`src/visualization/eda.py`): seaborn distributions.
4. **Model** (`src/models/`): scaled KNN and a decision tree.
5. **Evaluate** (`src/visualization/performance.py`): confusion matrices, ROC curves, metrics.

## Run

```bash
pip install -r requirements.txt
python main.py
```

`notebooks/exploration.ipynb` is the early exploration.
