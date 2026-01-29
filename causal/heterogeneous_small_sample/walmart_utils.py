import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

from pathlib import Path
import os, sys

module_dpath = Path(os.path.abspath(""))
repo_dpath = module_dpath.parent.parent
dsets_dpath = repo_dpath / 'datasets'

def load_walmart_dset(pareto=False):
    '''Load the walmart dataset and choose to add a pareto sales column or not'''
    # load the original dset
    df = pd.read_csv(dsets_dpath / 'Walmart.csv')
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

    # if I don't want a pareto sales column, just return original
    if not pareto:
        return df
    
    #### pareto
    # can make it more heterogenous by dropping `a` the pareto shape parameter 
    # (also make sure to rescale so the dollar amounts aren't weird) - 0.5 goes well with 1e12 for example and looks like 
    # state distributions (ie CA much bigger and NY and TX close together but also separate from rest or something) 
    # but that's all just empiracally playing around
    min_val = df.weekly_sales.quantile(0.1)
    a = 0.5 # Pareto shape parameter
    rescale_multiplier = 1e12 # rescale multiplier puts in similar dollars scale as true data
    # play with these to make them be similar scales as true
    df['pareto_sales'] = np.round(min_val / df.weekly_sales ** (1/a) * rescale_multiplier, 2) 
    return df



########################################################
############## ttest and cuped #########################  
########################################################


class SimpleExpData():
    '''Collapse the pre and post treatment down to a single number
    by aggregating and collapsing the time columns (so only one row per store)'''
    def __init__(self, 
                 df, 
                 sales_col='weekly_sales', 
                 intervention_day_str = "2012-09-07",
                 pre_treatment_start_day_str = "2012-06-08"
                ):
        # store args
        self.df, self.sales_col = df, sales_col
        self.intervention_day_str, self.pre_treatment_start_day_str = intervention_day_str, pre_treatment_start_day_str

        # collapse temporal dimension
        self.exp_df = self.collapse_time_dimension()
        return

    def collapse_time_dimension(self):
        '''creates the experimental dataset'''
        exp_df = pd.DataFrame(index=self.df.store.unique())
        exp_df.index.name = 'store'

        pre_treat_mask = (self.df.date >= self.pre_treatment_start_day_str) & (self.df.date < self.intervention_day_str)
        exp_df['pre_sales'] = self.df.loc[pre_treat_mask].groupby('store')[self.sales_col].sum()

        post_treat_mask = (self.df.date >= self.intervention_day_str)
        exp_df['raw_post_sales'] = self.df.loc[post_treat_mask].groupby('store')[self.sales_col].sum()

        return exp_df.reset_index()
        
    def designate_treatment_groups(self, percent_treated=0.5):
        '''Adds / updates a T column to self.df'''
        unique_stores = self.exp_df['store'].unique()
        n_stores = len(unique_stores)
        treated_stores = np.random.choice(unique_stores, 
                                          size=int(n_stores * percent_treated), 
                                          replace=False)

        self.exp_df['T'] = self.exp_df['store'].isin(treated_stores).astype(int)
        return
    
    def add_effect(self,
                   effect_params = (0.05, 0.02) # mean, std for percent effect
                   ):
        # np.random.seed(42)  # reproducibility
        mu, std = effect_params
        
        # simulate the effect and mask for only those with T=1 after the treatment date
        effect = np.random.normal(loc=mu, scale=std, size=len(self.exp_df))
        self.exp_df['effect'] = 0.0
        # apply effect only to treated units after intervention
        mask = (self.exp_df['T'] == 1)
        self.exp_df.loc[mask, 'effect'] = effect[mask]

        # adjust weekly sales
        self.exp_df['adj_post_sales'] = self.exp_df['raw_post_sales'] * (1 + self.exp_df['effect'])
        return
    
    def run_t_test_mc(self,
               effect_params = (0.05, 0.02), # mean, std for percent effect
               n_iters = 1000,
               percent_treated=0.5,
               alpha = 0.05
               ):
        outs = []
        for _ in range(n_iters):
            # split and add an effect
            self.designate_treatment_groups(percent_treated=percent_treated)
            self.add_effect(effect_params=effect_params)

            # split into treated and control
            treated = self.exp_df.loc[self.exp_df['T']==1, 'adj_post_sales']
            control = self.exp_df.loc[self.exp_df['T']==0, 'adj_post_sales']

            # run ttest
            t_stat, p_val = stats.ttest_ind(treated, control, equal_var=False)

            # store output
            outs.append((t_stat, p_val))
        mc_df = pd.DataFrame(outs, columns=['ttsat','pval'])
        prop_reject = (mc_df.pval < alpha).sum() / n_iters
        return mc_df, prop_reject


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