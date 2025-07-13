import numpy as np
import optuna
import time
from xgboost import XGBRegressor, XGBClassifier

# optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostOptunaTuner:
    def __init__(self, X_train, y_train, X_valid, y_valid, problem_type='regression'):
        """
        Parameters:
        - problem_type: 'regression' or 'binary'
        """
        assert problem_type in ['regression', 'binary'], "Only 'regression' and 'binary' are supported."
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.problem_type = problem_type

        # parameters that are constant across all 
        self.base_params = { 
            "booster": "gbtree",
            "verbosity": 0,
            "enable_categorical": True,
            "tree_method": "hist"
        }
        
        # update the objective based on problem type
        if self.problem_type == "regression":
            self.base_params["objective"] = "reg:squarederror"
        elif self.problem_type == "binary":
            self.base_params["objective"] = "binary:logistic"
        else:
            raise ValueError(f"Invalid problem type: '{self.problem_type}'")

        return

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def log_loss(y_true, y_pred, eps=1e-15):
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    def _objective(self, trial):
        # prep parameters
        tune_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 3, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            "early_stopping_rounds":25,
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
            return self.rmse(self.y_valid, preds)

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
            return self.log_loss(self.y_valid, proba)
        
    def optimize(self, total_tune_time=30, print_secs=10):
        '''Search for a certain amount of time (as opposed to the defaul which is n_trials)'''

        # prep for printing
        start, last = time.time(), time.time()

        # prep study
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)

        # train a new model while we're still under the alloted time
        while (time.time() - start) < total_tune_time:
            self.study.optimize(self._objective, n_trials=1)

            # logic to print every 10 sec with a status update on how long its been training for
            now = time.time()
            if now - last > print_secs:
                last = now
                print(f"Elapsed: {np.round(now-start)}sec | Iteration #{len(self.study.trials)}")
        
        # save the study attrs
        self.tuned_params = self.study.best_params
        self.training_params = {**self.base_params, **self.tuned_params}
        return