import numpy as np
import pandas as pd

from os import path, makedirs
from glob import glob
from tqdm import tqdm
from datetime import datetime
from utils import uni_distribution
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# === loading
len_train = 201917

top_folder = './CV_LB_log'
feats_folder = './raw_feature/selector-null importance-v4'
data_folder = './raw_feature'

today = datetime.today()
now = today.strftime('%m%d-%H%M')

# ===
train = pd.read_csv('./input/train.csv', usecols=['card_id', 'target'])
test = pd.read_csv('./input/test.csv', usecols=['card_id'])
raw = pd.concat([train, test], axis=0, sort=False)

files = glob('./CV_LB_log/Bayesian_oof/*.csv')

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

# ========= average all oof
raw['oof'] = raw.drop(columns=['card_id', 'target']).mean(axis=1)
train = raw[:len_train]
cv_score = mean_squared_error(train['oof'], train['target']) ** 0.5
print("CV score: {:<8.5f}".format(cv_score))
sub_folder = path.join(top_folder, 'average-CV-' + str(np.round(cv_score, 5)) + '_' + now)
makedirs(sub_folder, exist_ok=True)
raw[['card_id', 'oof']].to_csv(path.join(sub_folder, 'oof_average.csv'), index=False)
del raw['oof']

# ========= stacking
train = raw[:len_train]
test = raw[len_train:]

train = uni_distribution(train, 'target')

x_train = train.drop(columns=['card_id', 'target'])
y_train = train['target']
x_test = test.drop(columns=['card_id', 'target'])

folds = KFold(n_splits=6, shuffle=False, random_state=None)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
    val_data, val_y = x_train.iloc[val_idx], y_train.iloc[val_idx]

    reg = Ridge(alpha=1)
    reg.fit(trn_data, trn_y)

    oof[val_idx] = reg.predict(val_data)
    predictions += reg.predict(x_test) / folds.n_splits

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
