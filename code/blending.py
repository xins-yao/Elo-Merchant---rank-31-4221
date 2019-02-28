import numpy as np
import pandas as pd

from os import path, makedirs
from tqdm import tqdm
from datetime import datetime
from utils import uni_distribution
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.isotonic import IsotonicRegression

# === loading
len_train = 201917
min_target = -33.21928095
max_target = 17.9650684

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'

today = datetime.today()
now = today.strftime('%m%d-%H%M')

# === oof
train = pd.read_csv('./input/train.csv', usecols=['card_id', 'target'])
test = pd.read_csv('./input/test.csv', usecols=['card_id'])
raw = pd.concat([train, test], axis=0, sort=False)

prefix_old = 'F:/python/CreditCardLoyality/CV_LB_log/'
prefix_dataset_2 = 'F:/python/Elo/dateset2-reverse normalization - clip at 2.259 - auth - new/CV_LB_log/'
prefix_dataset_4 = 'F:/python/Elo/dataset4-uauth/CV_LB_log/'
prefix_all_feature = 'F:/python/Elo/all_feature/CV_LB_log/'

files = [
    # dataset 4
    # lgb-reg
    prefix_dataset_4 + 'v39-3.63147_0222-2012-tuned-based on v38/oof.csv',
    # lgb-clean
    prefix_dataset_4 + 'clean-v2-1.54431_0225-2119-remove oof-baseline/oof.csv',
    # lgb-clf
    prefix_dataset_4 + 'CV-0.09907_0216-1049/oof.csv',
    # xgb-reg
    prefix_dataset_4 + 'v36-3.63738_0216-1436-xgb-based on v35/oof.csv',
    # average
    prefix_dataset_4 + 'average-CV-3.62847_0222-2105/oof.csv',
    # stack
    prefix_dataset_4 + 'stacking-CV-3.63225_0225-2158/oof.csv',

    # dataset 3
    # lgb-reg
    './CV_LB_log/v41-3.6308_0222-2209-add credit_2-based on v40/oof.csv',
    # lgb-clean
    './CV_LB_log/clean-v1-1.55273_0216-1712/oof.csv',
    # lgb-clf
    './CV_LB_log/clf-outlier-v1-0.09909_0210-2319/oof.csv',
    # xgb-reg
    './CV_LB_log/v36-3.63761_0216-1312-xgb-based on v25/oof.csv',
    # average
    './CV_LB_log/average-v1-3.62933_0226-0835/oof.csv',
    # stack
    './CV_LB_log/stacking-v38-3.63537_0226-0835-random seed/oof.csv',

    # dataset 2
    # lgb-reg
    prefix_dataset_2 + 'v6-3.63619/oof.csv',
    # xgb-reg
    prefix_dataset_2 + 'v4-3.64319_0216-0014-xgb-based on v3/oof.csv',
    # average
    prefix_dataset_2 + 'average-v1-3.63438_0226-1929/oof.csv',
    # stack
    prefix_dataset_2 + 'stacking-v1-3.64176_0226-1929/oof.csv',

    # all_feature
    # lgb-reg
    prefix_all_feature + 'v8-3.63051_0225-1941-tuned/oof.csv',
    # lgb-clean
    prefix_all_feature + 'clean-v2-1.54382_0224-2245-remove oof/oof.csv',
    # lgb-clf
    prefix_all_feature + 'clf-outlier-v2-0.90627_0225-2152-remove oof/oof.csv',
    # average
    prefix_all_feature + 'average-v3-3.62804_0225-2210-random seed/oof.csv',
    # stack
    prefix_all_feature + 'stacking-v3-3.62854_0225-2210-random seed/oof.csv',
]

# files = [
#     './CV_LB_log/stacking-v37-3.61841-3.673_0225-2100-NO raw feature-NO stack bayesian/oof.csv',
# ]

# files = pd.read_csv('files.csv')['file']

count = 0
for f in tqdm(files):
    oof = pd.read_csv(f, usecols=['card_id', 'oof'])
    oof.columns = ['card_id', 'oof_' + str(count)]
    raw = pd.merge(raw, oof, how='left', on='card_id')
    count += 1

# raw.to_csv('merging.csv', index=False)

# ========= average all oof
# raw['oof'] = raw.drop(columns=['card_id', 'target']).mean(axis=1)
# train = raw[:len_train]
# cv_score = mean_squared_error(train['oof'], train['target']) ** 0.5
# print("CV score: {:<8.5f}".format(cv_score))
# sub_folder = path.join(top_folder, 'average-CV-' + str(np.round(cv_score, 5)) + '_' + now)
# makedirs(sub_folder, exist_ok=True)
# raw[['card_id', 'oof']].to_csv(path.join(sub_folder, 'oof_average.csv'), index=False)
# del raw['oof']

# === raw feature
if False:
    # === dataset 4
    feats_ds2 = [
        'card_id',

        'category_1_1_purchase_amount_min_new_to_hist',
        'new_month_12_purchase_amount_max',
        'new_month_1_purchase_amount_mean_to_std',
    ]
    raw_ds2 = pd.read_csv(
        'F:/python/Elo/dateset2-reverse normalization - clip at 2.259 - auth - new/raw_feature/raw_fe-v1.csv',
        usecols=feats_ds2
    )
    raw_ds2 = raw_ds2[feats_ds2]
    feats_ds2 = ['ds4_' + col for col in raw_ds2.columns.values if col != 'card_id']
    raw_ds2.columns = ['card_id'] + feats_ds2
    raw = pd.merge(raw, raw_ds2, how='left', on='card_id')
    del raw_ds2

    # === dataset old
    feats_old = [
        'card_id',

        'new_installments_1_purchase_amount_min',
    ]
    raw_old = pd.read_csv('F:/python/CreditCardLoyality/raw_final_not_fill_na.csv', usecols=feats_old)
    raw_old = raw_old[feats_old]
    feats_old = ['old_' + col for col in raw_old.columns.values if col != 'card_id']
    raw_old.columns = ['card_id'] + feats_old
    raw = pd.merge(raw, raw_old, how='left', on='card_id')
    del raw_old

    # === dataset 4
    feats_ds4 = [
        'card_id',

        'uauth_merchant_category_id_repurchase_mean',
        'hist_year_nunique',
        'new_month_10_purchase_amount_min_to_max',
        'hist_month_lag_-9_purchase_amount_sum',
        'hist_purchase_amount_mean_lag3_to_lag12',
        'days_feature2',
        'purchase_amount_count_per_month_new_to_hist',
    ]
    raw_ds4 = pd.read_csv(path.join(data_folder, 'raw_fe-v35-uauth.csv'), usecols=feats_ds4)
    raw_ds4 = raw_ds4[feats_ds4]
    feats_ds4 = ['ds4_' + col for col in raw_ds4.columns.values if col != 'card_id']
    raw_ds4.columns = ['card_id'] + feats_ds4
    raw = pd.merge(raw, raw_ds4, how='left', on='card_id')
    del raw_ds4

    # === dataset 3
    feats_ds3 = [
        'card_id',

        'month_1_purchase_amount_count_hist_m_new',
        'month_1_purchase_amount_mean_hist_m_new',
        'installments_1.0_purchase_amount_min_hist_m_new',
    ]
    raw_ds3 = pd.read_csv(path.join(data_folder, 'raw_fe-v20-add holiday feature.csv'), usecols=feats_ds3)
    raw_ds3 = raw_ds3[feats_ds3]
    feats_ds3 = ['ds3_' + col for col in raw_ds3.columns.values if col != 'card_id']
    raw_ds3.columns = ['card_id'] + feats_ds3
    raw = pd.merge(raw, raw_ds3, how='left', on='card_id')
    del raw_ds3

# === Fill NA
raw = raw.replace(np.inf, np.nan)
raw = raw.replace(-np.inf, np.nan)
raw = raw.fillna(0)

# === training
train = raw[:len_train]
test = raw[len_train:]

train = uni_distribution(train, 'target')

x_train = train.drop(columns=['card_id', 'target'])
y_train = train['target']
x_test = test.drop(columns=['card_id', 'target'])

print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))

folds = KFold(n_splits=6, shuffle=False, random_state=None)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
    val_data, val_y = x_train.iloc[val_idx], y_train.iloc[val_idx]

    reg = BayesianRidge()
    # reg = Ridge(alpha=1)
    reg.fit(trn_data, trn_y)

    oof[val_idx] = reg.predict(val_data)
    predictions += reg.predict(x_test) / folds.n_splits

oof = np.clip(oof, a_min=min_target, a_max=max_target)
predictions = np.clip(predictions, a_min=min_target, a_max=max_target)

cv_score = mean_squared_error(oof, y_train) ** 0.5
print("CV score: {:<8.5f}".format(cv_score))

# === output
sub_folder = path.join(top_folder, 'stacking-CV-' + str(np.round(cv_score, 5)) + '_' + now)
makedirs(sub_folder, exist_ok=True)

test['target'] = predictions
test[['card_id', 'target']].to_csv(path.join(sub_folder, 'submit.csv'), index=False)
train['target'] = oof

df_oof = pd.concat([train[['card_id', 'target']], test[['card_id', 'target']]], axis=0)
df_oof.columns = ['card_id', 'oof']

df_oof.to_csv(path.join(sub_folder, 'oof.csv'), index=False)
