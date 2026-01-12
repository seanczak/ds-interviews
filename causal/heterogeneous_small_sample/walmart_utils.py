import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


class TreatedDF():
    def __init__(self, 
                 df, 
                 sales_col='weekly_sales', 
                 intervention_day_str = "2012-06-08",
                 effect_params = (0.05, 0.02) # mean, std for percent effect
                ):
        # store args
        self.df, self.sales_col, self.effect_params = df, sales_col, effect_params
        self.intervention_date = pd.to_datetime(intervention_day_str)
        # add T column to df
        self.designate_treatment_groups()
        # add the effect in place (new col adj_sales)
        self.add_effect()
        
    def designate_treatment_groups(self):
        unique_stores = self.df['store'].unique()
        treated_stores = np.random.choice(unique_stores, size=len(unique_stores)//2, replace=False)

        self.df['T'] = self.df['store'].isin(treated_stores).astype(int)
        return
    
    def add_effect(self):
        # np.random.seed(42)  # reproducibility
        mu, std = self.effect_params
        
        # simulate the effect and mask for only those with T=1 after the treatment date
        effect = np.random.normal(loc=mu, scale=std, size=len(self.df))
        self.df['effect'] = 0.0
        mask = (self.df['date'] >= self.intervention_date) & (self.df['T'] == 1)
        # apply effect only to treated units after intervention
        self.df.loc[mask, 'effect'] = effect[mask]

        # adjust weekly sales
        self.df['adj_sales'] = self.df[self.sales_col] * (1 + self.df['effect'])
        return


class MCSimulation(TreatedDF):
    def __init__(self,
                 df, 
                 sales_col='weekly_sales', 
                 intervention_day_str = "2012-06-08",
                 effect_params = (0.05, 0.02), # mean, std for percent effect
                 pre_or_post = 'pre', # pre intervention for AA and post for AB
                ):
        # store parameters using inheritance
        super().__init__(df, 
                         sales_col=sales_col, 
                         intervention_day_str = intervention_day_str,
                         effect_params = effect_params)
        # also want to know if its pre or post 
        self.pre_or_post = pre_or_post
        return
    
    def generate_single_run_results(self):
        run = TreatedDF(self.df, effect_params=self.effect_params) # mu, std of effect --> 100% + N(mu,std)
        
        # only want the either pre or post experiment data for now
        if self.pre_or_post == 'pre':
            exp_df = run.df.loc[run.df['date'] < run.intervention_date]
        else:
            exp_df = run.df.loc[run.df['date'] >= run.intervention_date]

        # split into treated and 
        treated = exp_df.loc[exp_df['T'] == 1].groupby(['store'])['adj_sales'].sum()
        control = exp_df.loc[exp_df['T'] == 0].groupby(['store'])['adj_sales'].sum()
        
        t_stat, p_val = stats.ttest_ind(treated, control, equal_var=False)
        return t_stat, p_val
    
    def run_many_simulations(self, n_sim=1000):
        output = []
        for run in range(n_sim):
            t_stat, p_val = self.generate_single_run_results()
            output.append([run, t_stat, p_val])
        out_df = pd.DataFrame(output, columns = ['run_id','tstat','pval'])
        return out_df