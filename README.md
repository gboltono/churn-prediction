# Customer Churn Prediction

My final project for MAT 311 Intro to Data Science (Fall 2025). It's an end-to-end Python pipeline that predicts whether a customer will churn. I built it for a Kaggle customer churn prediction competition using 300K+ customer records.

## Results

Both models were scored on a 30% held-out validation set (90,938 customers):

| Model | ROC AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| **KNN (k = 10, scaled)** | **0.921** | **0.757** | 0.742 | 0.907 |
| Decision tree | 0.821 | 0.750 | 0.746 | 0.896 |

KNN was the stronger model. It ranks customers by churn risk far better than the decision tree (0.92 vs 0.82 AUC) at about the same F1.

## What the pipeline does

1. **Cleaning** (`src/data/preprocessing.py`)
   - Removes the `Customer Status` column, which leaks the answer.
   - Fills missing `Tenure` and `Last Interaction` with the mode and missing `Support Calls` with 0.
   - Fills in missing `Payment Delay` using the last due date and last payment date.
2. **Feature engineering** (`src/features/build_features.py`): one-hot encodes `Subscription Type` and `Contract Length` and maps `Gender` to 0/1.
3. **EDA** (`src/visualization/eda.py`): seaborn distributions of tenure, usage, spend and gender.
4. **Modeling** (`src/models/`): KNN with standard scaling in a scikit-learn `Pipeline`, plus a decision tree.
5. **Evaluation** (`src/visualization/performance.py`): confusion matrices, ROC curves, and the metrics in the table above.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

`main.py` reads `data/raw/train.csv` and `data/raw/test.csv`. It then shows the EDA plots, confusion matrices and ROC curves, and prints the validation metrics.

`notebooks/exploration.ipynb` is the notebook I used to explore the data before writing the pipeline.
