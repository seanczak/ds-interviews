from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd


class ClassificationEvals:
    """retention model is a binary classification - this is for scoring and tuning it individually (pre hurdle)"""

    def __init__(self, trues, preds):  # if true, labels need to be included
        # scores
        self.f1 = f1_score(trues, preds)
        self.r = recall_score(trues, preds)
        self.p = precision_score(trues, preds)
        self.a = accuracy_score(trues, preds)
        return

    def assemble_results_series(self):
        return pd.Series({"f1": self.f1, "recall": self.r, "precision": self.p, "accuracy": self.a})

    def assemble_results_df(
        self,
        index_name: str = None,
    ):
        """single row df"""
        return pd.DataFrame(self.assemble_results_series()).T

    def display_evals(self):
        print(
            f"""
F1 : {np.round(self.f1,4)}
recall : {np.round(self.r,4)}
precision: {np.round(self.p,4)}
accuracy : {np.round(self.a,4)}
"""
        )
        return


class ContinuousEvals:
    """Hurdle model is regression, so we need the following"""

    def __init__(self, trues, preds):
        self.trues, self.preds = trues, preds

        # save these so we can look at the distributions in case they are interesting
        self.errs = trues - preds

        # calc evals
        self.rmse = np.sqrt(np.mean(self.errs ** 2))
        self.mae = np.mean(abs(self.errs))
        # normalize them
        self.rmse_over_mean = self.rmse / self.trues.mean()
        self.mae_over_mean = self.mae / self.trues.mean()

        self.ave_bias = (preds - trues).mean()
        self.total_bias = (preds - trues).sum()
        return

    def calc_mae(self, ser):
        return



    def assemble_results_series(self):
        return pd.Series(
            {
                "rmse": self.rmse,
                "rmse_over_mean": self.rmse_over_mean,
                "mae": self.mae,
                "mae_over_mean": self.mae_over_mean,
                "ave_bias": self.ave_bias,
                "total_bias": self.total_bias,
            }
        )

    def assemble_results_df(
        self,
        index_name: str = None,
    ):
        """single row df"""
        return pd.DataFrame(self.assemble_results_series()).T