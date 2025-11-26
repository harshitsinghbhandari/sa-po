import numpy as np
import pandas as pd
def clean_data():
    pass
def get_data():
    clean_data()
    df = pd.read_csv('nse_data.csv',index_col=0)
    # df = df.tail(100)
    # first_int = df.iloc[0].astype(int)//2
    # df = df - first_int
    print(df.head())
    
    log_returns = np.log(df / df.shift(1)).dropna()

    cov_matrix = log_returns.cov()
    mean_log_returns = log_returns.mean().to_numpy()
    
    # print(log_returns)
    

    cov_matrix_array = cov_matrix.to_numpy()
    # return df
    return mean_log_returns, cov_matrix_array
# get_data()
