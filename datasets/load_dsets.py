from pathlib import Path
import pandas as pd
import numpy as np
import os

module_dpath = Path(os.path.abspath(""))
repo_dpath = module_dpath.parent
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