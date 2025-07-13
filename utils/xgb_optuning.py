import numpy as np
import pandas as pd
import optuna
import time
from xgboost import XGBRegressor, XGBClassifier

from .loss_functions import calc_log_loss, calc_rmse


class XGBoostOptunaTuner:
    """
    given the dsets (train and valid), and the type of problem (regression, binary classification)
    set up a study to tune for a set amount of time 
    for now parameter ranges are hard encoded, this wouldn't be ideal for a true pipeline

    Parameters:
    - problem_type: 'regression' or 'binary'
    - optuna_verbosity: set whether we want optuna to print every trial output or not
    """

    def __init__(self, 
                 X_train: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 X_valid: pd.DataFrame,
                 y_valid: pd.DataFrame, 
                 optuna_verbosity='low', # [low, high]
                 problem_type='regression' # [regression, binary]
                 ):
        # check inputs
        assert problem_type in ['regression', 'binary'], "Unrecognized problem_type"
        assert optuna_verbosity in ['low', 'high'], "Unrecognized optuna verbosity"
        
        # set verbosity
        if optuna_verbosity=='low':
            # pretty much no printing
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            # will print every trial result
            optuna.logging.set_verbosity(optuna.logging.INFO)
              

        # store attributes
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.problem_type = problem_type

        # some parameters that are constant across all 
        self.base_params = { 
            "booster": "gbtree",
            "verbosity": 0,
            "enable_categorical": True,
            "tree_method": "hist" # hist required for early stopping functionality with sklearn
        }
        
        # update the objective based on problem type
        if self.problem_type == "regression":
            self.base_params["objective"] = "reg:squarederror"
        elif self.problem_type == "binary":
            self.base_params["objective"] = "binary:logistic"
        
        # prep study
        sampler = optuna.samplers.TPESampler(seed=42) # set a seed for the default sampler
        self.study = optuna.create_study(direction='minimize', sampler=sampler)
        return

    def _objective(self, trial):
        '''Set up objective for either regression or binary classification'''
        
        # prep parameters
        tune_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 3, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            "early_stopping_rounds":25, # stop when valid score stops improving (reg term)
        }
        params = {**self.base_params, **tune_params}

        # objective for regression
        if self.problem_type == "regression":
            params["eval_metric"] = "rmse"
            model = XGBRegressor(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_valid, self.y_valid)],
                verbose=False
            )
            preds = model.predict(self.X_valid)
            return calc_rmse(self.y_valid, preds)

        # objective for binary classification
        elif self.problem_type == "binary":
            params["eval_metric"] = "logloss"
            model = XGBClassifier(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_valid, self.y_valid)],
                verbose=False
            )
            proba = model.predict_proba(self.X_valid)[:, 1]
            return calc_log_loss(self.y_valid, proba)
        
    def tune_for_nsecs(self, total_tune_time=30, print_secs=10):
        '''Search for a certain amount of time (as opposed to the defaul which is n_trials)
        
        This is split off from the study initiation so that we can tune longer on same study if desired'''

        # prep for printing
        start, last = time.time(), time.time()

        # tune for a certain amount of time (kind of a hack to get optuna to do this)
        while (time.time() - start) < total_tune_time:
            # run a single trial per loop pass
            self.study.optimize(self._objective, n_trials=1)

            # make a printout of status
            now = time.time()
            if now - last > print_secs:
                last = now
                print(f"Elapsed: {np.round(now-start)}sec | Total Iterations: {len(self.study.trials)}")

        # save new tuned params
        self.tuned_params = self.study.best_params
        self.training_params = {**self.base_params, **self.tuned_params}
        return