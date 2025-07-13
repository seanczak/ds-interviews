import pandas as pd
from sklearn.preprocessing import StandardScaler

from .train_test_split import train_valid_test_split

class LinearDataset:
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

        # determine the cat cols and the numeric cols in the features
        self.determine_cat_and_num_features()
        
        return
    
    def determine_cat_and_num_features(self):
        '''these will be used later for preprocessing conditions'''
        self.cat_cols = []
        self.num_cols = []
        for col, dtype in self.df[self.feat_cols].dtypes.items():
            # check if categorical
            if dtype in ["object", "string", "category"]:
                self.cat_cols.append(col)
            
            # elif numeric
            elif dtype in ["int64", "float64", "int32", "float32", "bool"]:
                self.num_cols.append(col)
            
            # else raise a type error on that column
            else:
                raise TypeError(f"Unsupported dtype: {dtype}")
        return
    
    def one_hot_encoding(self, drop_first=False):
        '''one-hot encodes categorical columns using training set only, and applies it to valid and test sets.
        - categorical_cols (list): list of categorical column names
        - drop_first (bool): whether to drop the first dummy to avoid multicollinearity (not a problem with regularization)'''
            
        # One-hot encode cat cols
        X_train_enc = pd.get_dummies(self.X_train, columns=self.cat_cols, drop_first=drop_first)
        X_valid_enc = pd.get_dummies(self.X_valid, columns=self.cat_cols, drop_first=drop_first)
        X_test_enc  = pd.get_dummies(self.X_test,  columns=self.cat_cols, drop_first=drop_first)

        # Align columns: add missing cols in valid/test sets, fill with 0
        self.encoded_cols = X_train_enc.columns
        X_valid_enc = X_valid_enc.reindex(columns=self.encoded_cols, fill_value=0)
        X_test_enc = X_test_enc.reindex(columns=self.encoded_cols, fill_value=0)

        if self.verbose: 
            print(f'\n{len(self.cat_cols)} categorical variables encoded')

        return X_train_enc, X_valid_enc, X_test_enc
    
    def standardize_numeric_features(self): 
        '''this is done in place'''

        # standardize the numeric cols
        scaler = StandardScaler()
        self.X_train[self.num_cols] = scaler.fit_transform(self.X_train[self.num_cols])
        self.X_valid[self.num_cols] = scaler.transform(self.X_valid[self.num_cols])
        self.X_test[self.num_cols] = scaler.transform(self.X_test[self.num_cols])

        if self.verbose: 
            print(f'\n{len(self.num_cols)} numeric variables standardized')
        return
    
    def impute_missing_values(self):
        ''' imputation values from training set (save the values) '''

        if self.verbose: 
            print('\nImputing missing values with mean and mode')

        # Impute numeric features with train-set means
        self.num_means = self.X_train[self.num_cols].mean()

        self.X_train.loc[:, self.num_cols] = self.X_train[self.num_cols].fillna(self.num_means)
        self.X_valid.loc[:, self.num_cols] = self.X_valid[self.num_cols].fillna(self.num_means)
        self.X_test.loc[:, self.num_cols]  = self.X_test[self.num_cols].fillna(self.num_means)

        # Impute categorical features with train-set modes
        self.cat_modes = self.X_train[self.cat_cols].mode().iloc[0]

        self.X_train.loc[:, self.cat_cols] = self.X_train[self.cat_cols].fillna(self.cat_modes)
        self.X_valid.loc[:, self.cat_cols] = self.X_valid[self.cat_cols].fillna(self.cat_modes)
        self.X_test.loc[:, self.cat_cols]  = self.X_test[self.cat_cols].fillna(self.cat_modes)

        return
    
    def prep_for_tuning(self,
                        split_type:str, # ['random', 'temporal']
                        time_col=None, # only used if temporal splitting
                        p_train=0.8, 
                        p_valid=0.1,
                        missing_val_impute=True,
                        ):
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
        if self.verbose:
            print('Train/valid/test split complete:')
            print(f'\t{len(self.y_train)} rows train, \n\t{len(self.y_valid)} rows valid, \n\t{len(self.y_test)} rows test')

        # missing values 
        # (can either: drop rows, impute from mean or mode)
        if missing_val_impute:
            self.impute_missing_values()
        # note I could also drop a full feature OR drop some rows before putting into here (like if a target was missing)
        # and impute the rest
        else:
            if self.verbose: print('\nDropping missing values rows')
            self.X_train = self.X_train.dropna()
            self.X_valid = self.X_valid.dropna()
            self.X_test  =  self.X_test.dropna()

        # encode cats for each X
        self.X_train,self.X_valid, self.X_test = self.one_hot_encoding(drop_first=False)

        # standardize numeric features
        self.standardize_numeric_features()

        # new feature list (now that we've potentially one hot encoded)
        self.orig_feats = self.feat_cols
        self.feat_cols = list(self.encoded_cols)

        return

    
    def prep_for_final_training(self):
        '''combine the train and valid and save to self'''
        self.X_trainvalid = pd.concat([self.X_train, self.X_valid])
        self.y_trainvalid = pd.concat([self.y_train, self.y_valid])
        if self.verbose:
            print('\nTrain and valid set combined for final training:')
            print(f'\t{len(self.y_trainvalid)} rows in "trainvalid" dset')
        return