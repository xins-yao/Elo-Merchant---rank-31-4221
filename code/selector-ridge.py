import gc
import sys
import time
import warnings
import importlib
import numpy as np
import pandas as pd
import lightgbm as lgb

from os import path
from glob import glob
from tqdm import tqdm
from utils import uni_distribution

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import ridge

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
warnings.simplefilter('ignore', UserWarning)

# ================================================================================== Params
len_train = 201917

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'

# ================================================================================== Load Data
train = pd.read_csv('./input/train.csv', usecols=['card_id', 'target'])
test = pd.read_csv('./input/test.csv', usecols=['card_id'])
raw = pd.concat([train, test], axis=0, sort=False)
del train
del test

# === oof
prefix_old = 'F:/python/CreditCardLoyality/CV_LB_log/'
prefix_dataset_2 = 'F:/python/Elo/dateset2-reverse normalization - clip at 2.259 - auth - new/CV_LB_log/'
prefix_dataset_4 = 'F:/python/Elo/dataset4-uauth/CV_LB_log/'
prefix_all_feature = 'F:/python/Elo/all_feature/CV_LB_log/'

files = [
    './CV_LB_log/average-v2-3.6226_0226-1947-average all oof/oof.csv',
]

count = 0
for f in tqdm(files):
    oof = pd.read_csv(f, usecols=['card_id', 'oof'])
    oof.columns = ['card_id', 'oof_' + str(count)]
    raw = pd.merge(raw, oof, how='left', on='card_id')
    count += 1

raw = raw.fillna(0)
print('Load Data Done.')

# ==================================================================================
feats = [col for col in raw.columns.values if col not in ['card_id', 'first_active_month', 'target']]
categorical_feats = \
    [f for f in raw.columns if (raw[f].dtype == 'object' and f not in ['card_id', 'first_active_month'])]
categorical_feats.extend(['feature_1', 'feature_2', 'feature_3', 'hist_category_2_mode', 'new_category_2_mode'])
feats = [col for col in feats if col not in categorical_feats]

# ================================================================================== Training
train = raw[:len_train]
del raw

# === uniform distribution
train = uni_distribution(df=train, key='target')
y_train = train['target']
# y_train = (train['target'] < -33).astype(int)

model = ridge.Ridge(alpha=1)
folds = KFold(n_splits=6, shuffle=False, random_state=None)

col_all = [col for col in feats]
col_use = [
    'oof_4',
    # 'oof_2',
    # 'oof_4',
    # 'oof_6',
    # 'oof_8',
    # 'oof_9',
    # 'oof_11',
    # 'oof_13',
]
col_loop = [col for col in col_all if col not in col_use]
print(len(col_loop))

baseline = 100
min_score = 100
f_best = None
feature_to_select = 50
ls_score = list(np.zeros(len(col_use)))

while len(col_use) < feature_to_select:
    print('n_selected_feature: {}'.format(len(col_use)))

    # loop
    col_remove = []
    for f in tqdm(col_loop):
        x_train = train[col_use + [f]]
        cv = cross_validate(model, x_train, y_train, scoring='neg_mean_squared_error', cv=folds, n_jobs=12)
        score = (-cv['test_score'].mean()) ** 0.5
        if score < baseline:
            if score < min_score:
                min_score = score
                print('feature: {}, score: {}'.format(f, score))
                improve = True
                f_best = f
        else:
            col_remove.append(f)
    # update
    for f in col_remove:
        col_loop.remove(f)
    baseline = score

    # record
    if improve:
        col_use.append(f_best)
        ls_score.append(min_score)
        col_loop.remove(f_best)
        improve = False
    else:
        break

print(col_use)
# pd.DataFrame({'feature': col_use, 'score': ls_score}).to_csv('feature_selection_ridge.csv')

