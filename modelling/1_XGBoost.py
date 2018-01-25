import pandas as pd
#from mylib import *
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from pyarrow.feather import read_feather

#in this script I want to run an XGBoost on the last exposure date differences only
# I will train and test on 1:50  and use CV
def get_cols(df, stem):
    return [col for col in list(df) if stem in col]

#set parameters
filename = "F:\\Valentina\\Experiments\\exp_lookback\\extra_data\\st_combined_train_unmatched_test_capped_freq_datediff_alldays.feather"
last_exp_stem = 'LAST_EXP_'
label = "label"
pos_label = 1
neg_label = 0

#select columns from dataset
df = read_feather(filename)
exp_cols = get_cols(df, last_exp_stem)
df = df[exp_cols.append(label)]

#set 1:50 ratio
ratio = 50
n_pos = df[df[label] == pos_label].shape[0]
n_neg = df[df[label] == neg_label].shape[0]
n_sample_neg = n_pos *ratio
sampled_neg = df[df[label] == neg_label].sample(n= n_sample_neg, random_state = 2)
sampled_df = df[df[label] == pos_label].concat(sampled_neg)

#train and test the model
X = sampled_df[exp_cols]
Y = sampled_df[label]
model = XGBClassifier()
kfold = KFold(n_splits = 5, random_state = 7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

