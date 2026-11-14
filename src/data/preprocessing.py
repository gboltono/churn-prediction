#main.py - import libraries
import pandas as pd

from src.features.build_features import one_hot_encode, compute_payment_delay

def fill_na_with_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Fill missing values in the specified columns of the DataFrame with the mode."""
    for col in columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
    return df

def fill_cols_na_with_0s(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Fill missing values in the specified columns of the DataFrame with 0s."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def convert_data_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Convert columns to specified data types."""
    for column, dtype in column_types.items():
        df[column] = df[column].astype(dtype)
    return df

def clean_data(train: pd.DataFrame, test: pd.DataFrame):
    #create copies of original data
    train_clean = train.copy()
    test_clean = test.copy()

    #1-hot encode
    train_clean = one_hot_encode(train_clean, ['Subscription Type', 'Contract Length'])
    test_clean = one_hot_encode(test_clean, ['Subscription Type', 'Contract Length'])

    #map binary columns
    train_clean['Gender'] = train_clean['Gender'].map({'Male': 0, 'Female': 1})
    test_clean['Gender'] = test_clean['Gender'].map({'Male': 0, 'Female': 1})

    #remove customer status to prevent data leakage
    train_clean = train_clean.drop(columns=['Customer Status'])
    test_clean = test_clean.drop(columns=['Customer Status'])

    #fill with mode
    train_clean = fill_na_with_mode(train_clean, ['Tenure', 'Last Interaction'])
    test_clean = fill_na_with_mode(test_clean, ['Tenure','Last Interaction'])

    #fill Support Calls with zeros
    train_clean = fill_cols_na_with_0s(train_clean, ['Support Calls'])
    test_clean = fill_cols_na_with_0s(test_clean, ['Support Calls'])

    #fill payment delay based on last due date and last payment date
    train_clean = compute_payment_delay(train_clean)
    test_clean = compute_payment_delay(test_clean)

    #drop date columns (since we already have payment delay)
    train_clean = train_clean.drop(columns=['Last Due Date', 'Last Payment Date'])
    test_clean = test_clean.drop(columns=['Last Due Date', 'Last Payment Date'])

    #train_clean['Spending Per Month'] = train_clean['Total Spend'] / train_clean['Tenure'].replace(0, 1)
    #test_clean['Spending Per Month'] = test_clean['Total Spend'] / test_clean['Tenure'].replace(0, 1)
    return train_clean, test_clean