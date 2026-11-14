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
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

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

#print validation metrics
for name, y_pred, y_prob in [("KNN", y_val_pred_knn, val_prob_knn), ("Decision Tree", y_val_pred_dt, val_prob_dt)]:
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    print(f"{name}: ROC AUC {roc_auc_score(y_val, y_prob):.3f}, F1 {f1_score(y_val, y_pred):.3f}, "
          f"sensitivity {tp / (tp + fn):.3f}, specificity {tn / (tn + fp):.3f}")

#plot ROC Curve
plot_roc_curve(y_val, val_prob_knn, val_prob_dt)

print("Testing Models...")
y_test_pred_knn = knn_model.predict(test_clean.drop(columns=['CustomerID']))
test_prob_knn = knn_model.predict_proba(test_clean.drop(columns=['CustomerID']))[:, 1]

y_test_pred_dt = dt_model.predict(test_clean.drop(columns=['CustomerID']))
test_prob_dt = dt_model.predict_proba(test_clean.drop(columns=['CustomerID']))[:, 1]

print("done.")