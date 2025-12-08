#main.py - import libraries
import pandas as pd

def load_data():
    test = pd.read_csv("data/raw/test.csv")
    train = pd.read_csv("data/raw/train.csv")
    return train, test
