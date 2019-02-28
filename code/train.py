import gc
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from os import path, makedirs
from glob import glob
from datetime import datetime, date
from tqdm import tqdm
from utils import Logger, uni_distribution

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
warnings.simplefilter('ignore', UserWarning)

# ================================================================================== Params
len_train = 201917
len_test = 123623
min_target = -33.21928095

# threshold = 0.35
# n_target_0_trans_0_train = 1513
# n_target_33_trans_0_train = 574
# n_target_33_ref_date_1_train = 555  # month_diff = 5
# n_target_33_ref_date_12_train = 315 # month_diff = 6
# n_target_0_trans_0_train_pred = n_target_0_trans_0_train * threshold
# n_target_33_trans_0_train_pred = n_target_33_trans_0_train * threshold
# n_target_33_ref_date_1_train_pred = n_target_33_ref_date_1_train * threshold
# n_target_33_ref_date_12_train_pred = n_target_33_ref_date_12_train * threshold
# n_target_0_trans_0_test_pred = threshold * n_target_0_trans_0_train * len_test / len_train
# n_target_33_trans_0_test_pred = threshold * n_target_33_trans_0_train * len_test / len_train
# n_target_33_ref_date_1_test_pred = threshold * n_target_33_ref_date_1_train * len_test / len_train
# n_target_33_ref_date_12_test_pred = threshold * n_target_33_ref_date_12_train * len_test / len_train

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'

today = datetime.today()
now = today.strftime('%m%d-%H%M')
log_name = now + '.txt'
sys.stdout = Logger(path.join(top_folder, log_name))

# ================================================================================== Feature Selection
# data v9
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

# print('kmeans feature')
# raw_kmeans = pd.read_csv('./raw_feature/raw_kmeans.csv', usecols=['card_id', 'cluster', 'cluster_mean_target'])
# raw = pd.merge(raw, raw_kmeans, how='left', on='card_id')
# feats.extend(['cluster', 'cluster_mean_target'])

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

# === category
categorical_feats = \
    [f for f in raw.columns if (raw[f].dtype == 'object' and f not in ['card_id', 'first_active_month'])]
categorical_feats.extend(['feature_1', 'feature_2', 'feature_3', 'hist_category_2_mode', 'new_category_2_mode', 'cluster'])
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

# col_save = feats.copy()
# col_save.extend(['card_id', 'target'])
# raw[col_save].to_csv('./raw_feature/raw_fe-v42.csv', index=False)

# ================================================================================== Training
train = raw[:len_train]
test = raw[len_train:]

# === train without outlier
# train = raw[raw['target'] > -33]
# test = raw[~(raw['target'] > -33)]
# print(len(test))

del raw

# === uniform distribution
train = uni_distribution(df=train, key='target')

# === training
x_train = train[feats]
y_train = train['target']
x_test = test[feats]

param = {
    'objective': 'regression',
    'num_leaves': 180,
    'min_data_in_leaf': 49,
    'max_depth': 9,
    'learning_rate': 0.01,
    'lambda_l1': 4.69291359115505,
    'lambda_l2': 0.2769777415493315,
    "boosting": "gbdt",
    "feature_fraction": 0.259170147130108,
    'bagging_freq': 8,
    "bagging_fraction": 0.9340833266631288,
    "metric": 'rmse',
    "verbosity": -1,
    'n_jobs': 12,
    "random_state": 2019

    # befor knn
    # 'objective': 'regression',
    # 'num_leaves': 64,
    # 'min_data_in_leaf': 25,
    # 'max_depth': 9,
    # 'learning_rate': 0.01,
    # 'lambda_l1': 0.43187736213573746,
    # 'lambda_l2': 2.7757122571321697,
    # "boosting": "gbdt",
    # "feature_fraction": 0.39676098987612096,
    # 'bagging_freq': 8,
    # "bagging_fraction": 0.8608155689585768,
    # "metric": 'rmse',
    # "verbosity": -1,
    # 'n_jobs': 12,
    # "random_state": 2019

    # 'objective': 'regression',
    # 'num_leaves': 83,
    # 'min_data_in_leaf': 43,
    # 'max_depth': 8,
    # 'learning_rate': 0.01,
    # 'lambda_l1': 2.676,
    # 'lambda_l2': 2.179,
    # "boosting": "gbdt",
    # "feature_fraction": 0.394,
    # 'bagging_freq': 8,
    # "bagging_fraction": 0.9793,
    # "metric": 'rmse',
    # "verbosity": -1,
    # 'n_jobs': 12,
    # "random_state": 2019
}

# folds = KFold(n_splits=5, shuffle=True, random_state=15)
folds = KFold(n_splits=6, shuffle=False, random_state=None)
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, train['outlier'].values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(x_train.iloc[trn_idx],
                           label=y_train.iloc[trn_idx],
                           categorical_feature=cat_feats)
    val_data = lgb.Dataset(x_train.iloc[val_idx],
                           label=y_train.iloc[val_idx],
                           categorical_feature=cat_feats)
    num_round = 10000
    reg = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof[val_idx] = reg.predict(x_train.iloc[val_idx], num_iteration=reg.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = x_train.columns
    fold_importance_df["importance"] = reg.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += reg.predict(x_test, num_iteration=reg.best_iteration) / folds.n_splits

cv_score = mean_squared_error(oof, y_train) ** 0.5
print("CV score: {:<8.5f}".format(cv_score))

# ==================================================================== blending
# n_new_trans = \
#     pd.read_csv('./raw_feature/raw_single_new_purchase_amount.csv', usecols=['card_id', 'new_purchase_amount_count'])
# n_new_trans = n_new_trans.fillna(0)
# month_diff = pd.read_csv('./raw_feature/raw_single_hist_month_diff.csv')
#
# # ================== train
# df_train = pd.DataFrame(train['card_id'])
# df_train['target1'] = oof
# df_train['target2'] = oof
# df_train = pd.merge(df_train, n_new_trans, how='left', on='card_id')
# df1 = df_train[df_train['new_purchase_amount_count'] == 0]
# df1 = df1.sort_values(by='target2', ascending=True)
# idx1 = df1[:int(n_target_33_trans_0_train_pred)].index
# # df_train.loc[idx, 'target2'] = min_target
# # === month_diff
# df_train = pd.merge(df_train, month_diff, how='left', on='card_id')
# df2 = df_train[df_train['hist_month_diff_min'] == 5]
# df2 = df2.sort_values(by='target2', ascending=True)
# idx2 = df2[:int(n_target_33_ref_date_1_train_pred)].index
#
# idx = np.unique(np.concatenate([idx1, idx2]))
# df_train.loc[idx, 'target2'] = min_target
# oof = (df_train['target1'] * 0.8 + df_train['target2'] * 0.2).values
#
# # ================== test
# df_test = pd.DataFrame(test['card_id'])
# df_test['target1'] = predictions
# df_test['target2'] = predictions
# df_test = pd.merge(df_test, n_new_trans, how='left', on='card_id')
# df1 = df_test[df_test['new_purchase_amount_count'] == 0]
# df1 = df1.sort_values(by='target2', ascending=True)
# idx1 = df1[:int(n_target_33_trans_0_test_pred)].index
# # === month_diff
# df_test = pd.merge(df_test, month_diff, how='left', on='card_id')
# df2 = df_test[df_test['hist_month_diff_min'] == 5]
# df2 = df2.sort_values(by='target2', ascending=True)
# idx2 = df2[:int(n_target_33_ref_date_1_test_pred)].index
#
# idx = np.unique(np.concatenate([idx1, idx2]))
# df_test.loc[idx, 'target2'] = min_target
# predictions = (df_test['target1'] * 0.8 + df_test['target2'] * 0.2).values
#
# # === score
# cv_score_blend = mean_squared_error(oof, y_train) ** 0.5
# print("CV score after blending: {:<8.5f}".format(cv_score_blend))

# ================================================================================== Save Data
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
