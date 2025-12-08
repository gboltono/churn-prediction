import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df : pd.DataFrame):
    #exploratory plots:
    
    #Histplot Tenure
    sns.histplot(x = "Tenure", data = df,bins = 20)
    plt.title('Tenure Distribution')
    plt.show()

    sns.histplot(x = "Usage Frequency", data = df,bins = 20)
    plt.title('Usage Frequency Distribution')
    plt.show()

    sns.histplot(x = "Total Spend", data = df,bins = 20)
    plt.title('Total Spend Distribution')
    plt.show()

    sns.histplot(x = "Gender", data = df,bins = 2)
    plt.title('Gender Distribution (0 = male, 1 = female)')
    plt.show()
    '''
    sns.countplot(x = "Total Spend", data = df)
    plt.title('Total Spend Distribution for ' + traintest)
    plt.show()
    '''
