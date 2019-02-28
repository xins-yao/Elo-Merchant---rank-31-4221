import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import uni_distribution

from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)

# ================================================================================== Params
len_train = 201917

# ================================================================================== Load Data
df_feats = pd.read_csv('./raw_feature/columns_and_null_count.csv')
feats = [col for col in df_feats['feature'] if 'purchase_date' in col]

col_read = feats.copy()
col_read.append('target')
raw = pd.read_csv('raw_fe.csv', usecols=col_read)
print('Load Data Done.')

# ================================================================================== Main
train = raw[:len_train]
train = train.replace(np.inf, np.nan)
train = train.replace(-np.inf, np.nan)
train = train.fillna(0)

# === uniform distribution
train = uni_distribution(df=train, key='target')

# === training
x_train = train[feats]
y_train = train['target']

param = {
    'objective': 'regression',
    'num_leaves': 80,
    'min_data_in_leaf': 25,
    'max_depth': 7,
    'learning_rate': 0.01,
    'lambda_l1': 0.13,
    "boosting": "gbdt",
    "feature_fraction": 0.85,
    'bagging_freq': 8,
    "bagging_fraction": 0.9,
    "metric": 'rmse',
    "verbosity": -1,
    'n_jobs': 12,
    "random_state": 2019
}
model = lgb.LGBMRegressor(**param)
folds = KFold(n_splits=5, shuffle=True, random_state=15)
selector = RFECV(model, step=1, min_features_to_select=1, cv=folds, scoring='neg_mean_squared_error', verbose=1, n_jobs=6)
selector.fit(x_train, y_train)

# ================================================================================== Save Result
result = pd.DataFrame({'feature': x_train.columns.values, 'rank': selector.ranking_})
result.to_csv('result_RFECV.csv', index=False)
