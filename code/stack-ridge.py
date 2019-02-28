import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd

from os import path
from glob import glob
from tqdm import tqdm
from scipy.stats import ks_2samp
from utils import uni_distribution

from sklearn.metrics import mean_squared_error
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

# ========= dataset old
feats_old = [
    # 'new_installments_1_purchase_amount_min',
    # 'new_category_1_0_purchase_amount_mean',
]
col_read = feats_old.copy()
col_read.append('card_id')

# === temp
raw_old = pd.read_csv('F:/python/CreditCardLoyality/raw_final_not_fill_na.csv')
raw_old = raw_old.drop(columns=['target'])

# raw_old = pd.read_csv('F:/python/CreditCardLoyality/raw_final_not_fill_na.csv', usecols=col_read)
# feats_old = ['old_' + col for col in raw_old.columns.values if col != 'card_id']
# raw_old.columns = ['card_id'] + feats_old
#
raw = pd.merge(raw, raw_old, how='left', on='card_id')
del raw_old

# ========= dataset 2
feats_ds2 = [
    'category_1_1_purchase_amount_min_new_to_hist',
    'new_month_12_purchase_amount_max',
    'new_month_1_purchase_amount_mean_to_std',
]
col_read = feats_ds2.copy()
col_read.append('card_id')

# === temp
# raw_ds2 = pd.read_csv(
#     'F:/python/Elo/dateset2-reverse normalization - clip at 2.259 - auth - new/raw_feature/raw_fe-v1.csv')
# raw_ds2 = raw_ds2.drop(columns=['target'])

raw_ds2 = pd.read_csv(
    'F:/python/Elo/dateset2-reverse normalization - clip at 2.259 - auth - new/raw_feature/raw_fe-v1.csv',
    usecols=col_read
)
feats_ds2 = ['ds2_' + col for col in raw_ds2.columns.values if col != 'card_id']
raw_ds2.columns = ['card_id'] + feats_ds2

raw = pd.merge(raw, raw_ds2, how='left', on='card_id')
del raw_ds2

# ========= dataset 3
feats_ds3 = [
    'month_1_purchase_amount_count_hist_m_new',
    'month_1_purchase_amount_mean_hist_m_new',
    'installments_1.0_purchase_amount_min_hist_m_new',
]
col_read = feats_ds3.copy()
col_read.append('card_id')

# === temp
# raw_ds3 = pd.read_csv(path.join(data_folder, 'raw_fe-v20-add holiday feature.csv'))
# raw_ds3 = raw_ds3.drop(columns=['target'])

raw_ds3 = pd.read_csv(path.join(data_folder, 'raw_fe-v20-add holiday feature.csv'), usecols=col_read)
feats_ds3 = ['ds3_' + col for col in raw_ds3.columns.values if col != 'card_id']
raw_ds3.columns = ['card_id'] + feats_ds3
# #
raw = pd.merge(raw, raw_ds3, how='left', on='card_id')
del raw_ds3

# ========= dataset 4
feats_ds4 = [
    'uauth_merchant_category_id_repurchase_mean',
    'hist_year_nunique',
    'new_month_10_purchase_amount_min_to_max',
    'hist_month_lag_-9_purchase_amount_sum',
    'hist_purchase_amount_mean_lag3_to_lag12',
    'days_feature2',
    'purchase_amount_count_per_month_new_to_hist',
]
col_read = feats_ds4.copy()
col_read.append('card_id')

# === temp
# raw_ds4 = pd.read_csv(path.join(data_folder, 'raw_fe-v35-uauth.csv'))
# raw_ds4 = raw_ds4.drop(columns=['target'])

raw_ds4 = pd.read_csv(path.join(data_folder, 'raw_fe-v35-uauth.csv'), usecols=col_read)
feats_ds4 = ['ds4_' + col for col in raw_ds4.columns.values if col != 'card_id']
raw_ds4.columns = ['card_id'] + feats_ds4
#
raw = pd.merge(raw, raw_ds4, how='left', on='card_id')
del raw_ds4

# === oof
files = [
    './CV_LB_log/average-v2-3.6226_0226-1947-average all oof/oof.csv',
]

count = 0
for f in tqdm(files):
    oof = pd.read_csv(f, usecols=['card_id', 'oof'])
    oof.columns = ['card_id', 'oof_' + str(count)]
    raw = pd.merge(raw, oof, how='left', on='card_id')
    count += 1

raw = raw.replace(np.inf, np.nan)
raw = raw.replace(-np.inf, np.nan)
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
test = raw[len_train:]
del raw

# === remove imbalance feature
list_p_value = []
for i in tqdm(feats):
    list_p_value.append(ks_2samp(test[i], train[i])[1])
Se = pd.Series(list_p_value, index=feats).sort_values()
list_discarded = list(Se[Se < .1].index)
print(list_discarded)
for col in tqdm(list_discarded):
    feats.remove(col)

# === uniform distribution
train = uni_distribution(df=train, key='target')
y_train = train['target']

train = train[feats]
test = test[feats]
gc.collect()

model = ridge.Ridge(alpha=1)
folds = KFold(n_splits=6, shuffle=False, random_state=None)

col_all = [col for col in feats]
col_use = [
    'oof_0',
]
col_use.extend(feats_old)
col_use.extend(feats_ds2)
col_use.extend(feats_ds3)
col_use.extend(feats_ds4)
col_loop = [col for col in col_all if col not in col_use]
print(len(col_loop))

# baseline
x_train = train[col_use]
cv = cross_validate(model, x_train, y_train, scoring='neg_mean_squared_error', cv=folds, n_jobs=12)
baseline = max_score = cv['test_score'].mean()

min_gain = 0.00005
f_best = None
feature_to_select = 50
ls_score = list(np.zeros(len(col_use)-1)) + [baseline]

while len(col_use) < feature_to_select:
    print('n_selected_feature: {}'.format(len(col_use)))
    col_remove = []
    # run through all feature and find the one improve cv the most
    for f in tqdm(col_loop):
        x_train = train[col_use + [f]]
        cv = cross_validate(model, x_train, y_train, scoring='neg_mean_squared_error', cv=folds, n_jobs=12)
        score = cv['test_score'].mean()
        if score > (baseline + min_gain):
            if score > max_score:
                max_score = score
                print('feature: {}, score: {}'.format(f, score))
                f_best = f
        else:
            col_remove.append(f)

    if max_score <= baseline + min_gain * 10:
        break
    baseline = max_score

    # remove useless feature
    for f in col_remove:
        col_loop.remove(f)
    col_loop.remove(f_best)

    # recording
    col_use.append(f_best)
    ls_score.append(max_score)

pd.DataFrame({'feature': col_use, 'score': ls_score}).to_csv('feature_selection_ridge.csv')


