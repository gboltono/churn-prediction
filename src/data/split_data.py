import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df):
    y = df['Churn']
    X = df.drop(columns=['Churn', 'CustomerID'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_val, y_train, y_val