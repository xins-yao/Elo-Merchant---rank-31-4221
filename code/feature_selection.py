import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from os import path
from sklearn.metrics import mean_squared_error

gc.enable()
warnings.simplefilter('ignore', UserWarning)

# ================================================================================== Params
len_train = 201917

# ================================================================================== Load Data

# ===
# feats_folder = './raw_feature/selector-null importance-v4'
# corr_scores_df = pd.read_csv(path.join(feats_folder, 'corr_scores.csv'))
# scores_df = pd.read_csv(path.join(feats_folder, 'scores.csv'))
#
# threshold = -1
# feats = [col for col in corr_scores_df.loc[corr_scores_df['split_score'] > threshold, 'feature']]
# feats = [col for col in scores_df.loc[scores_df['split_rank'] < 1200, 'feature'] if col in feats]
# feats = [col for col in scores_df.loc[scores_df['gain_rank'] < 1200, 'feature'] if col in feats]
# print(len(feats))
#
# # ================================================================================== Load Data
# col_read = feats.copy()
# col_read.extend(['card_id', 'target'])
#
# raw = pd.read_csv('raw_fe.csv', usecols=col_read)
# print('Load Data Done.')
#
# df_minus = pd.read_csv('./raw_feature/minus_columns_and_null_count.csv')
# col_minus = [col for col in df_minus['feature'].values if 'hist_m_new' in col]
# raw_minus = pd.read_csv('./raw_feature/raw_fe_minus.csv', usecols=col_minus)
# raw[col_minus] = raw_minus

# ===
# data v9
feats_folder = './raw_feature/selector-null importance-v4'
corr_scores_df = pd.read_csv(path.join(feats_folder, 'corr_scores.csv'))
scores_df = pd.read_csv(path.join(feats_folder, 'scores.csv'))

threshold = 99
feats = [col for col in corr_scores_df.loc[corr_scores_df['split_score'] > threshold, 'feature']]
feats = [col for col in scores_df.loc[scores_df['split_rank'] < 250, 'feature'] if col in feats]
feats = [col for col in scores_df.loc[scores_df['gain_rank'] < 250, 'feature'] if col in feats]
print(len(feats))

col_df = pd.read_csv('uauth_columns_and_null_count.csv')
col_uauth = [col for col in col_df['feature'] if 'uauth' in col]
feats.extend(col_uauth)
print(len(feats))

feats = [col for col in feats if col in col_df['feature'].values]
print(len(feats))

# ================================================================================== Load Data
col_read = feats.copy()
col_read.extend(['card_id', 'target'])

raw = pd.read_csv('raw_fe_uauth.csv', usecols=col_read)
data = raw[:len_train]

# =======================
categorical_feats = \
    [f for f in data.columns if (data[f].dtype == 'object' and f not in ['card_id', 'first_active_month'])]
categorical_feats.extend(
    ['feature_1', 'feature_2', 'feature_3', 'new_category_2_mode', 'hist_category_2_mode', 'uauth_category_2_mode'])
categorical_feats = [col for col in categorical_feats if col in data.columns]

for f_ in categorical_feats:
    data[f_], _ = pd.factorize(data[f_])
    data[f_] = data[f_].astype('category')


# =======================
def get_feature_importances(data, shuffle, seed=None):
    train_features = [f for f in data if f not in ['target', 'card_id', 'first_active_month']]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['target'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['target'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 12
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(data[train_features]))

    return imp_df


# =======================
# Seed the unexpected randomness of this world
np.random.seed(2019)
# Get the actual importance, i.e. without shuffling
# actual_imp_df = get_feature_importances(data=data, shuffle=False)
# actual_imp_df.to_csv('actual_importances_distribution_rf.csv', index=False)

actual_imp_df = pd.DataFrame()
nb_runs = 50
seeds = []
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importance
    seed = np.random.randint(2019)
    seeds.append(seed)
    imp_df = get_feature_importances(data=data, shuffle=False, seed=seed)
    imp_df['run'] = i + 1
    # Concat the latest importance with the old ones
    actual_imp_df = pd.concat([actual_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)
actual_imp_df.to_csv('actual_importances_distribution_rf.csv', index=False)


# =======================
null_imp_df = pd.DataFrame()
nb_runs = 100

start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importance
    imp_df = get_feature_importances(data=data, shuffle=True)
    imp_df['run'] = i + 1
    # Concat the latest importance with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)
null_imp_df.to_csv('null_importances_distribution_rf.csv', index=False)


# # =======================
# actual_imp_df = pd.read_csv('actual_importances_distribution_rf.csv',
#                             usecols=['feature', 'importance_gain', 'importance_split', 'trn_score'])
# null_imp_df = pd.read_csv('null_importances_distribution_rf.csv',
#                           usecols=['feature', 'importance_gain', 'importance_split', 'trn_score', 'run'])

# # =======================
feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
scores_df.to_csv('scores.csv', index=False)

# =======================
correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
corr_scores_df.to_csv('corr_scores.csv', index=False)


# =======================
# def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
#     # Fit LightGBM
#     dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
#     lgb_params = {
#         'objective': 'regression',
#         'boosting_type': 'gbdt',
#         'learning_rate': .1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'num_leaves': 31,
#         'max_depth': -1,
#         'seed': 13,
#         'n_jobs': 12,
#         'min_split_gain': .00001,
#         'reg_alpha': .00001,
#         'reg_lambda': .00001,
#         'metric': 'rmse',
#     }
#
#     # Fit the model
#     hist = lgb.cv(
#         params=lgb_params,
#         train_set=dtrain,
#         num_boost_round=2000,
#         categorical_feature=cat_feats,
#         nfold=5,
#         stratified=False,
#         shuffle=True,
#         early_stopping_rounds=50,
#         verbose_eval=0,
#         seed=17
#     )
#     # Return the last mean / std values
#     return hist['rmse-mean'][-1], hist['rmse-stdv'][-1]
#
#
# # =======================
# for threshold in [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]:
#     split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
#     split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
#     gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
#     gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
#
#     print('Results for threshold %3d' % threshold)
#     split_results = score_feature_selection(df=data, train_features=split_feats, cat_feats=split_cat_feats,
#                                             target=data['target'])
#     print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
#     gain_results = score_feature_selection(df=data, train_features=gain_feats, cat_feats=gain_cat_feats,
#                                            target=data['target'])
#     print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
#
# print(seeds)
