import gc
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from os import path
from utils import uni_distribution, Logger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

gc.enable()
warnings.simplefilter('ignore', UserWarning)
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)

# ================================================================================== Params
len_train = 201917
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'
sys.stdout = Logger('bayes_log.txt')

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
print(len(feats))

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

# === category
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

# ==================================================================================
train = raw[:len_train]
test = raw[len_train:]
del raw

train = uni_distribution(df=train, key='target')

x_train = train[feats]
y_train = train['target']
x_test = test[feats]


def lgb_eval(num_leaves, min_data_in_leaf, max_depth, lambda_l1, lambda_l2, feature_fraction, bagging_fraction):
    param = {
        'objective': 'regression',
        'learning_rate': 0.01,
        "boosting": "gbdt",
        "metric": 'rmse',
        "verbosity": -1,
        'bagging_freq': 1,
        # "random_state": 2019,
        "random_state": None,

        "num_leaves": int(round(num_leaves)),
        'min_data_in_leaf': int(round(min_data_in_leaf)),
        'max_depth': int(round(max_depth)),
        'lambda_l1': max(lambda_l1, 0),
        'lambda_l2': max(lambda_l2, 0),
        'feature_fraction': max(min(feature_fraction, 1), 0),
        'bagging_fraction': max(min(bagging_fraction, 1), 0),
    }
    print(param)
    # folds = KFold(n_splits=6, shuffle=False, random_state=15)
    folds = KFold(n_splits=5, shuffle=True, random_state=None)

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
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
                        verbose_eval=0,
                        early_stopping_rounds=200)
        oof[val_idx] = reg.predict(x_train.iloc[val_idx], num_iteration=reg.best_iteration)
        predictions += reg.predict(x_test, num_iteration=reg.best_iteration) / folds.n_splits
    cv_score = -(mean_squared_error(oof, y_train) ** 0.5)

    test['oof'] = predictions
    train['oof'] = oof
    df_oof = pd.concat([train[['card_id', 'oof']], test[['card_id', 'oof']]], axis=0)
    df_oof.to_csv('./CV_LB_log/Bayesian_oof/oof-' + str(np.round(cv_score, 7)) + '.csv', index=False)

    return cv_score


param_grid = {
    'num_leaves': (64, 256),
    'min_data_in_leaf': (25, 50),
    'max_depth': (5, 8.99),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 3),
    'feature_fraction': (0.1, 0.9),
    'bagging_fraction': (0.8, 1),
}

lgbBO = BayesianOptimization(lgb_eval, param_grid, random_state=0)
lgbBO.maximize(init_points=50, n_iter=150)
param_opt = lgbBO.max['params']
print(param_opt)

df = pd.DataFrame({
    'params': list(param_opt.keys()),
    'value': list(param_opt.values()),
})
df.to_csv('best_params.csv')
