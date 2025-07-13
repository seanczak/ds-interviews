import pandas as pd

def fit_categorical_encoder(X_train: pd.DataFrame) -> dict:
    """Fit categorical encodings to X_train and grab the mappings for them"""
    cat_mappings = {} 
    for col, dtype in X_train.dtypes.items():
        if dtype in ["object", "string", "category"]:
            categories = X_train[col].astype('category').cat.categories
            cat_mappings[col] = categories
    return cat_mappings # {col:categories}

def apply_categorical_encoding(X: pd.DataFrame, cat_mappings: dict) -> pd.DataFrame:
    """ apply categorical mapping from training to a feature set X """
    X = X.copy()
    for col, dtype in cat_mappings.items():
        X[col] = pd.Categorical(X[col], categories=cat_mappings[col])
    return X