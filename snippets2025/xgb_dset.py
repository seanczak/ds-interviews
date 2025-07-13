import pandas as pd

from pandas_cat_encoder import fit_categorical_encoder, apply_categorical_encoding
from train_test_split import train_valid_test_split


class XgboostDataset:
    """
    - Accepts a df and encodes the features so its ready for analysis and/or training
    - Allows for feature list to be inferred by colums that aren't either index cols or cols to be dropped
    - option to hold onto some id columns that won't be used in training but will be preserved for analysis
    """

    def __init__(
        self,
        df: pd.DataFrame,
        targ_col: str,
        feat_cols=[],  # if empty list then we will assume all non identifier and non targ col columns are features
        drop_cols=[],  # those that are neither targs or feats (or index)
        id_cols=[],  # these will remain as index columns (not included in model but not dropped)
        verbose=False,
    ):

        # store args as attrs
        self.feat_cols, self.targ_col, self.drop_cols, self.id_cols = feat_cols, targ_col, drop_cols, id_cols
        self.verbose = verbose

        # this will be the main attr of the class
        self.df = df.drop(columns=drop_cols) 

        # update feat cols if its empty list "[]" (otherwise this does nothing)
        if self.feat_cols == []:
            self.feat_cols = [col for col in df.columns if col not in self.id_cols + self.drop_cols + [self.targ_col]]

        
        return
    
    def prep_for_tuning(self,
                        split_type: str, # ['random', 'temporal']
                        time_col=None,
                        p_train=0.8, 
                        p_valid=0.1):
        '''3 steps
            - splits data into train, valid and test 
            - save the id cols to the index of the df's (not used in training but could've been used for splitting)
            - then encodes the categoricals
        save everything to self'''
        
        # split into train, valid and test sets
        df_train, df_valid, df_test = train_valid_test_split(df=self.df, 
                                                             split_type=split_type,
                                                             time_col=time_col,
                                                             p_train=p_train, 
                                                             p_valid=p_valid)
        
        # save id cols to index (purposefully doing this after splitting in case time is included)
        if self.id_cols:
            df_train = df_train.set_index(self.id_cols)
            df_valid = df_valid.set_index(self.id_cols)
            df_test  =  df_test.set_index(self.id_cols)
        
        # Extract X and y
        self.X_train, self.y_train = df_train[self.feat_cols], df_train[self.targ_col]
        self.X_valid, self.y_valid = df_valid[self.feat_cols], df_valid[self.targ_col]
        self.X_test,  self.y_test  =  df_test[self.feat_cols],  df_test[self.targ_col]

        # encode cats for each X
        cat_mappings = fit_categorical_encoder(self.X_train)
        self.X_train = apply_categorical_encoding(self.X_train, cat_mappings)
        self.X_valid = apply_categorical_encoding(self.X_valid, cat_mappings)
        self.X_test  = apply_categorical_encoding(self.X_test,  cat_mappings)
        return
    
    def prep_for_final_training(self):
        '''combine the train and valid and save to self'''
        self.X_trainvalid = pd.concat([self.X_train, self.X_valid])
        self.y_trainvalid = pd.concat([self.y_train, self.y_valid])
        return