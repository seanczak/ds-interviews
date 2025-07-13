from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_log_loss(y_true, y_pred, eps=1e-15):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def ridge_grid_search(X_train, 
                      y_train, 
                      X_valid, 
                      y_valid, 
                      alpha_grid = np.logspace(-4, 2, 10)  # log space 10 values from 1e-4 to 1e2
                     ):
    best_alpha = None
    best_model = None
    best_score = np.inf

    results = []

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = calc_rmse(y_valid, preds)
        results.append((alpha, rmse))

        if rmse < best_score:
            best_score = rmse
            best_alpha = alpha
            best_model = model

    return best_model, best_alpha, best_score, results



def logistic_ridge_grid_search(X_train, 
                               y_train, 
                               X_valid, 
                               y_valid, 
                               C_grid = np.logspace(-4, 2, 10)  # from strong regularization to weak (1e-4 to 1e2)
                               ):
    ''' Note C = 1 / lambda so its inverted strength'''

    best_C = None
    best_model = None
    best_score = np.inf
    results = []

    for C in C_grid:
        model = LogisticRegression(C=C, penalty='l2', solver='liblinear')  # use liblinear for binary
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_valid)[:, 1]  # keep only P(y=1)
        loss = calc_log_loss(y_valid, probas)
        results.append((C, loss))

        if loss < best_score:
            best_score = loss
            best_C = C
            best_model = model

    return best_model, best_C, best_score, results