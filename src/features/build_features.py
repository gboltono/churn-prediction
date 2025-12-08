import pandas as pd
#import numpy as np

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode specified categorical columns."""
    return pd.get_dummies(df, columns=columns, dtype = int)


def compute_payment_delay(df):
    df = df.copy()

    # Convert MM-DD → datetime (using fixed year)
    df['DueDate_dt'] = pd.to_datetime("2024-" + df['Last Due Date'], format="%Y-%m-%d", errors="coerce")
    df['PayDate_dt'] = pd.to_datetime("2024-" + df['Last Payment Date'], format="%Y-%m-%d", errors="coerce")

    # Calculate difference in days
    df['ComputedDelay'] = (df['PayDate_dt'] - df['DueDate_dt']).dt.days

    # Fill Payment Delay ONLY if it is NaN
    df['Payment Delay'] = df['Payment Delay'].fillna(df['ComputedDelay'])

    # Drop helper columns
    df = df.drop(columns=['DueDate_dt', 'PayDate_dt', 'ComputedDelay'])

    return df