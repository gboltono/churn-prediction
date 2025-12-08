import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors=10):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    model.fit(X_train, y_train)
    return model