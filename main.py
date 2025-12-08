#main.py - import libraries
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocessing import clean_data
from src.data.split_data import split_data
from src.models.knn_model import train_knn_model
from src.models.decision_tree import train_decision_tree_model
from src.visualization.eda import plot_eda
from src.visualization.performance import plot_confusion_matrices
from src.visualization.performance import plot_roc_curve
from sklearn.metrics import confusion_matrix

#load data
print("loading data...")
train, test = load_data()
true_y = train['Churn']

print("cleaning data...")
train_clean, test_clean = clean_data(train, test)

print("Visualizing data...")
plot_eda(train_clean)

print("Splitting Data...")
X_train, X_val, y_train, y_val = split_data(train_clean)

print("training Models...")
knn_model = train_knn_model(X_train, y_train)
dt_model = train_decision_tree_model(X_train, y_train)

print("Evaluating on Evaluation Set...")
y_val_pred_knn = knn_model.predict(X_val)
y_val_pred_dt = dt_model.predict(X_val)

val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
val_prob_dt = dt_model.predict_proba(X_val)[:, 1]

#plot confusion matrices
plot_confusion_matrices(y_val, y_val_pred_knn, y_val_pred_dt)

#plot ROC Curve
plot_roc_curve(y_val, val_prob_knn, val_prob_dt)

print("Testing Models...")
y_test_pred_knn = knn_model.predict(test_clean.drop(columns=['CustomerID']))
test_prob_knn = knn_model.predict_proba(test_clean.drop(columns=['CustomerID']))[:, 1]

y_test_pred_dt = dt_model.predict(test_clean.drop(columns=['CustomerID']))
test_prob_dt = dt_model.predict_proba(test_clean.drop(columns=['CustomerID']))[:, 1]

print("done.")