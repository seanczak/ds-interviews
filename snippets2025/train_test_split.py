import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def random_row_split(df: pd.DataFrame, 
                     p_split=0.8):
    '''May need to do this twice so lets avoid calling them "train" or "test" just yet
    - p_split is for the "left" data set (eg 80% for training data)'''

    # train test split naively (80/20 no temporal considerations)
    df_left, df_right = train_test_split(df, train_size=p_split, random_state=42)
    return df_left, df_right


def temporal_row_split(df: pd.DataFrame, 
                       p_split=0.8, 
                       time_col=None):
    ''' Splits `df` into df_left and df_right based on a temporal ordering.
    - p_split: float in (0,1), fraction of data to go into df_left (e.g., 0.8 for 80%)
    - time_col: optional, name of time column to sort by. If None, use index order.
    '''

    if time_col is not None:
        df = df.sort_values(time_col).reset_index(drop=True)
    
    n = df.shape[0]
    n_left = int(np.floor(n * p_split))
    
    # split the data temporally - left is earlier and right is later data
    df_left = df.iloc[:n_left].copy()
    df_right = df.iloc[n_left:].copy()
    return df_left, df_right


def train_valid_test_split( df: pd.DataFrame, 
                            split_type: str, # ['random', 'temporal']
                            time_col=None,
                            p_train=0.8, 
                            p_valid=0.1):
    '''Splits data into 3 groups using either random or temporal splitting'''
    
    # check that there is enough for a test group
    p_test = 1 - p_train - p_valid
    assert p_test > 0, 'p_valid and p_train add up to greater than 1'
    # set up proportion for second split (splitting test and valid)
    p_second_split = p_valid / (p_valid + p_test)

    # split data based on split_type
    if split_type == 'random':
        df_train, df_test_valid = random_row_split(df, p_split=p_train)
        df_valid, df_test       = random_row_split(df_test_valid, p_split=p_second_split)

    elif split_type == 'temporal':
        df_train, df_test_valid = temporal_row_split(df, p_split=p_train, time_col=time_col)
        df_valid, df_test       = temporal_row_split(df_test_valid, p_split=p_second_split, time_col=time_col)

    else:
        raise ValueError(f"Invalid split type: '{split_type}'")

    return df_train, df_valid, df_test