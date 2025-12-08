import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrices(y_test, y_test_pred_knn, y_test_pred_dt):
    # Confusion Matrix for KNN
    cm_knn = confusion_matrix(y_test, y_test_pred_knn)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
    plt.title('KNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Confusion Matrix for Decision Tree
    cm_dt = confusion_matrix(y_test, y_test_pred_dt)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
    plt.title('Decision Tree Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, test_prob_knn, test_prob_dt):
    fpr_knn, tpr_knn, _ = roc_curve(y_test, test_prob_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    fpr_dt, tpr_dt, _ = roc_curve(y_test, test_prob_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)

    plt.figure()
    plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN ROC curve (area = %0.2f)' % roc_auc_knn)
    plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label='Decision Tree ROC curve (area = %0.2f)' % roc_auc_dt)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()