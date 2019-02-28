import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from os import path, makedirs
from datetime import datetime
from utils import Logger, uni_distribution

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
warnings.simplefilter('ignore', UserWarning)

# ================================================================================== Params
len_train = 201917

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'

today = datetime.today()
now = today.strftime('%m%d-%H%M')
log_name = now + '.txt'
sys.stdout = Logger(path.join(top_folder, log_name))

# ================================================================================== Feature Selection
corr_scores_df = pd.read_csv(path.join(feats_folder, 'corr_scores.csv'))
scores_df = pd.read_csv(path.join(feats_folder, 'scores.csv'))

threshold = 99
feats = [col for col in corr_scores_df.loc[corr_scores_df['split_score'] > threshold, 'feature']]
feats = [col for col in scores_df.loc[scores_df['split_rank'] < 250, 'feature'] if col in feats]
feats = [col for col in scores_df.loc[scores_df['gain_rank'] < 250, 'feature'] if col in feats]
print(len(feats))

# data v18
feats_folder = './raw_feature/selector-null importance-v18'
corr_scores_df = pd.read_csv(path.join(feats_folder, 'corr_scores.csv'))
scores_df = pd.read_csv(path.join(feats_folder, 'scores.csv'))

threshold = 99
feats_v18 = [col for col in corr_scores_df.loc[corr_scores_df['split_score'] > threshold, 'feature']]
feats_v18 = [col for col in scores_df.loc[scores_df['split_rank'] < 250, 'feature'] if col in feats_v18]
feats_v18 = [col for col in scores_df.loc[scores_df['gain_rank'] < 250, 'feature'] if col in feats_v18]

# combine
for col in feats_v18:
    if col not in feats:
        feats.append(col)

# ================================================================================== Load Data
col_read = feats.copy()
col_read.extend(['card_id', 'target'])

col_extend = [
    'hist_purchase_amount_mean',
    'new_purchase_amount_mean',
    'hist_purchase_amount_sum',
    'new_purchase_amount_sum',

    'hist_purchase_amount_count',
]
col_extend = [col for col in col_extend if col not in feats]
col_read.extend(col_extend)

raw = pd.read_csv(path.join(data_folder, 'raw_fe-v20-add holiday feature.csv'), usecols=col_read)
print('Load Data Done.')

print('knn feature')
raw_knn = pd.read_csv('./raw_feature/raw_date_knn_500.csv', usecols=['card_id', 'nn_500_mean_target'])
raw = pd.merge(raw, raw_knn, how='left', on='card_id')
feats.append('nn_500_mean_target')

# revisit_ratio
raw_revisit = pd.read_csv('./raw_feature/raw_single_auth_merchant_id.csv')
raw = pd.merge(raw, raw_revisit, how='left', on='card_id')
feats.extend([
    'auth_merchant_id_repurchase_ratio',
 ])

# credit_2
train = pd.read_csv('./input/train.csv', usecols=['card_id', 'feature_2'])
test = pd.read_csv('./input/test.csv', usecols=['card_id', 'feature_2'])
df = pd.concat([train, test], axis=0, sort=False)
raw = pd.merge(raw, df, how='left', on='card_id')
raw['credit_2'] = raw['feature_2'] * raw['hist_purchase_amount_count'] * raw['elapsed_time']
feats.append('credit_2')

# ==================================================================================
# feats = [col for col in raw.columns.values if col not in ['card_id', 'first_active_month', 'target']]

# === public regression data
raw_reg = pd.read_csv('./raw_feature/public/regression.csv')
raw_reg = raw_reg.drop(columns=['card_id'])
col_extend = [col for col in raw_reg.columns.values]
raw[col_extend] = raw_reg
for col in col_extend:
    if col not in feats:
        feats.append(col)

categorical_feats = \
    [f for f in raw.columns if (raw[f].dtype == 'object' and f not in ['card_id', 'first_active_month'])]
categorical_feats.extend(['feature_1', 'feature_2', 'feature_3', 'hist_category_2_mode', 'new_category_2_mode'])
cat_feats = [col for col in feats if col in categorical_feats]

for f_ in cat_feats:
    raw[f_], _ = pd.factorize(raw[f_])
    raw[f_] = raw[f_].astype('category')

# ================================================================================== Temporary
# repurchase amount
raw['hist_purchase_amount_mean'] = raw['hist_purchase_amount_mean']/100/0.00001503 + 500
raw['hist_repurchase_amount'] = raw['hist_purchase_amount_mean'] * raw['hist_merchant_id_repurchase_mean']
raw['hist_repurchase_amount'] = (raw['hist_repurchase_amount'] - 500) * 100 * 0.00001503
feats.append('hist_repurchase_amount')

raw['new_purchase_amount_sum'] = raw['new_purchase_amount_sum']/100/0.00001503 + 500
raw['new_repurchase_amount_sum'] = raw['new_purchase_amount_sum'] * raw['hist_merchant_id_repurchase_mean']
raw['new_repurchase_amount_sum'] = (raw['new_repurchase_amount_sum'] - 500) * 100 * 0.00001503
feats.append('new_repurchase_amount_sum')

# lag2 feature
hist = pd.read_csv('./raw_feature/raw_multi_hist_month_lag.csv')
raw['hist_purchase_amount_max_lag2'] = \
    hist[['hist_month_lag_0_purchase_amount_max', 'hist_month_lag_-1_purchase_amount_max']].max(axis=1)
raw['hist_purchase_amount_max_lag2'] = (raw['hist_purchase_amount_max_lag2'] - 500) * 100 * 0.00001503
feats.append('hist_purchase_amount_max_lag2')

# feats = np.unique(feats)
print(len(feats))

# ================================================================================== Training
train = raw[:len_train]
test = raw[len_train:]
del raw

# === uniform distribution
train = uni_distribution(df=train, key='target')

# === training
x_train = train[feats]
y_train = train['target']
x_test = test[feats]

param = {
    'objective': 'reg:linear',
    'num_leaves': 83,
    "min_child_weight": 43,
    'max_depth': 8,
    'learning_rate': 0.01,
    'reg_alpha': 2.676,
    'reg_lambda': 2.179,
    "booster": "gbtree",
    "subsample": 0.8,
    'colsample_bytree': 0.7,
    "metric": 'rmse',
    "silent": 1,
    'n_jobs': 12,
    "random_state": 2019
}

folds = KFold(n_splits=6, shuffle=False, random_state=None)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

test_data = xgb.DMatrix(x_test)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = xgb.DMatrix(x_train.iloc[trn_idx], y_train.iloc[trn_idx])
    val_data = xgb.DMatrix(x_train.iloc[val_idx], y_train.iloc[val_idx])

    num_round = 10000
    reg = xgb.train(param,
                    trn_data,
                    num_round,
                    evals=[(trn_data, 'train'), (val_data, 'test')],
                    verbose_eval=100,
                    early_stopping_rounds=100)
    oof[val_idx] = reg.predict(val_data, ntree_limit=reg.best_ntree_limit)

    dict_imp = reg.get_score(importance_type='gain')
    fold_importance_df = pd.DataFrame({'feature': list(dict_imp.keys()), 'importance': list(dict_imp.values())})
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += reg.predict(test_data, ntree_limit=reg.best_ntree_limit) / folds.n_splits

cv_score = mean_squared_error(oof, y_train) ** 0.5
print("CV score: {:<8.5f}".format(cv_score))

sub_folder = path.join(top_folder, 'CV-' + str(np.round(cv_score, 5)) + '_' + now)
makedirs(sub_folder, exist_ok=True)

feature_importance_df.to_csv(path.join(sub_folder, 'feat_imp-' + str(np.round(cv_score, 5)) + '.csv'))

# =======
test['target'] = predictions
test[['card_id', 'target']].to_csv(path.join(sub_folder, 'submit.csv'), index=False)
train['target'] = oof

df_oof = pd.concat([train[['card_id', 'target']], test[['card_id', 'target']]], axis=0)
df_oof.columns = ['card_id', 'oof']
df_oof.to_csv(path.join(sub_folder, 'oof.csv'), index=False)
