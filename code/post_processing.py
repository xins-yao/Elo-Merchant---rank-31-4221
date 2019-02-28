import gc
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from tqdm import tqdm
from os import path, makedirs
from datetime import datetime
from utils import Logger, uni_distribution

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
warnings.simplefilter('ignore', UserWarning)

# ================================================================================== Params
len_train = 201917

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'
oof_folder = './CV_LB_log/stacking-v37-3.61841-3.673_0225-2100-NO raw feature-NO stack bayesian'

# ================================================================================== Feature Selection
# using old v18
train = pd.read_csv('./input/train.csv', usecols=['card_id', 'target'])
oof = pd.read_csv(path.join(oof_folder, 'oof.csv'))
pred = pd.read_csv(path.join(oof_folder, 'oof-self.csv'))
# pred = pd.read_csv('./CV_LB_log/stacking-CV-3.62301_0220-0137/oof-old-clf.csv')


oof = pd.merge(oof, pred[['card_id', 'new_purchase_amount_count', 'pred', 'hist_month_diff_min']], how='left',
               on='card_id')
oof = pd.merge(oof, train, how='left', on='card_id')
print(oof.columns)

oof_train = oof[:len_train]
oof_test = oof[len_train:]

# ======================================================================================================= LR
# # find the best threshold
# min_target = -33.21928095
# lr = LinearRegression(fit_intercept=True, n_jobs=12)
# df = oof_train.sort_values(by='pred', ascending=True)
# # df = oof_train.sort_values(by='pred', ascending=False)
# baseline = mean_squared_error(df['oof'], df['target']) ** 0.5
# best_index = -1
# best_coef = []
# best_intercept = -1
# print('baseline: {}'.format(baseline))
#
# # |  0   |  1  |    2    |  3   |    4    |   5    |
# # |------|-----|---------|------|---------|--------|
# # | c_id | oof | n_trans | pred | ob_date | target |
#
# for i in tqdm(range(500)):
#     if df.iloc[i, 4] == 5:
#         df.iloc[i, 3] = min_target
#     else:
#         df.iloc[i, 3] = 0
#
#     lr.fit(df[['oof', 'pred']], df['target'])
#     blend = lr.predict(df[['oof', 'pred']])
#
#     current = mean_squared_error(blend, df['target']) ** 0.5
#     if current < baseline:
#         baseline = current
#         best_index = df.index[i]
#         best_coef = lr.coef_
#         best_intercept = lr.intercept_
#         print(i, current, df.index[i], lr.coef_, lr.intercept_)
#
# # test data post-processing
# df = oof.sort_values(by='pred', ascending=True)
# # df = oof.sort_values(by='pred', ascending=False)
# i = 0
# while df.index[i] != best_index:
#     if df.iloc[i, 4] == 5:
#         df.iloc[i, 3] = min_target
#     else:
#         df.iloc[i, 1] = 0.65 * df.iloc[i, 1]
#     i += 1
# df['oof'] = best_coef[0] * df['oof'] + best_coef[1] * df['pred'] + best_intercept
# print(i, best_coef, best_intercept)
#
# # ===
# test = pd.read_csv('./input/test.csv', usecols=['card_id'])
# test = pd.merge(test, df[['card_id', 'oof']], how='left', on='card_id')
# test.columns = ['card_id', 'target']
# test.to_csv(path.join(oof_folder, 'post_process.csv'), index=False)

# =================================================================== Manual threshold
# find the best threshold
min_target = -33.21928095
threshold = 0.845
df = oof_train.sort_values(by='pred', ascending=True)
# df = oof_train.sort_values(by='pred', ascending=False)
baseline = mean_squared_error(df['oof'], df['target']) ** 0.5
best_index = -1
print(baseline)
for i in range(2000):
    if df.iloc[i, 4] == 5:
        df.iloc[i, 1] = threshold * df.iloc[i, 1] + (1 - threshold) * min_target
    # else:
    #     df.iloc[i, 1] = 0.5 * df.iloc[i, 1]
    current = mean_squared_error(df['oof'], df['target']) ** 0.5
    if current < baseline:
        baseline = current
        best_index = df.index[i]
        print(i, current, df.index[i], df.iloc[i, 0])

# test data post-processing
df = oof.sort_values(by='pred', ascending=True)
i = 0
while df.index[i] != best_index:
    if df.iloc[i, 4] == 5:
        df.iloc[i, 1] = threshold * df.iloc[i, 1] + (1 - threshold) * min_target
    i += 1
print(i)

# ===
test = pd.read_csv('./input/test.csv', usecols=['card_id'])
test = pd.merge(test, df[['card_id', 'oof']], how='left', on='card_id')
test.columns = ['card_id', 'target']
test.to_csv('post_process.csv', index=False)
