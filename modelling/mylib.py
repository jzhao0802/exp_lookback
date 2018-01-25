import numpy as np

def cut_lookback(x, max_days):
    return x.where(x <= max_days, np.nan)

def get_cols(df, stem):
    return [col for col in list(df) if stem in col]