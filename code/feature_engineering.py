import gc
import time
import datetime
import numpy as np
import pandas as pd

gc.enable()
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)

# ================================================================================== Load Data
raw = pd.read_csv('./raw_feature/raw_concat.csv', parse_dates=['first_active_month'])
raw = raw.replace(np.inf, np.nan)
raw = raw.replace(-np.inf, np.nan)

# ================================================================================== Feature Fill NA Value
ref_date = datetime.date(2018, 5, 1)

nan_idx = 213495
raw.loc[nan_idx, 'first_active_month'] = \
    pd.to_datetime(ref_date - datetime.timedelta(days=int(raw.loc[nan_idx, 'hist_purchase_date_min'])))

raw['elapsed_time'] = (ref_date - raw['first_active_month'].dt.date).dt.days
raw['active_year'] = raw['first_active_month'].dt.year
raw['active_month'] = raw['first_active_month'].dt.month

# ================================================================================== Feature Combination
raw['days_feature1'] = raw['elapsed_time'] * raw['feature_1']
raw['days_feature2'] = raw['elapsed_time'] * raw['feature_2']
raw['days_feature3'] = raw['elapsed_time'] * raw['feature_3']

print('processing purchase_amount')
# === purchase_amount - new + hist
raw['purchase_amount_sum'] = raw['hist_purchase_amount_sum'] + raw['new_purchase_amount_sum']
raw['purchase_amount_count'] = raw['hist_purchase_amount_count'] + raw['new_purchase_amount_count']
raw['purchase_amount_mean'] = raw['purchase_amount_sum'] / raw['purchase_amount_count']
raw['purchase_amount_max'] = raw[['hist_purchase_amount_max', 'new_purchase_amount_max']].max(axis=1)
raw['purchase_amount_min'] = raw[['hist_purchase_amount_min', 'new_purchase_amount_min']].min(axis=1)

# === purchase_amount - hist / total
raw['hist_purchase_amount_sum_ratio'] = raw['hist_purchase_amount_sum'] / raw['purchase_amount_sum']
raw['hist_purchase_amount_mean_ratio'] = raw['hist_purchase_amount_mean'] / raw['purchase_amount_mean']
raw['hist_purchase_amount_count_ratio'] = raw['hist_purchase_amount_count'] / raw['purchase_amount_count']
raw['hist_purchase_amount_max_ratio'] = raw['hist_purchase_amount_max'] / raw['purchase_amount_max']
raw['hist_purchase_amount_min_ratio'] = raw['hist_purchase_amount_min'] / raw['purchase_amount_min']

# === per day / per month
for l in ['hist', 'new']:
    raw[l + '_purchase_amount_per_day'] = raw[l + '_purchase_amount_sum'] / raw[l + '_purchase_date_ptp']
    raw[l + '_purchase_amount_per_month'] = \
        raw[l + '_purchase_amount_sum'] / (raw[l + '_month_lag_max'] - raw[l + '_month_lag_min'] + 1)

    raw[l + '_purchase_amount_count_per_day'] = raw[l + '_purchase_amount_count'] / raw[l + '_purchase_date_ptp']
    raw[l + '_purchase_amount_count_per_month'] = \
        raw[l + '_purchase_amount_count'] / (raw[l + '_month_lag_max'] - raw[l + '_month_lag_min'] + 1)

# === transactions frequency
# raw.loc[raw['new_purchase_amount_count'] == 0, 'new_transactions_freq'] = 0
# raw['transactions_freq_new_to_hist'] = raw['new_transactions_freq'] / raw['hist_transactions_freq']

# === purchase_amount_sum - monthly & daily
dict_label_lags = {'hist': list(range(-13, 1)), 'new': list(range(1, 3))}
for func in ['sum', 'count']:
    for label, lags in dict_label_lags.items():
        cols = ['_'.join([label, 'month_lag', str(idx), 'purchase_amount_' + func]) for idx in lags]
        raw[label + '_monthly_purchase_amount_' + func + '_min'] = raw[cols].min(axis=1)
        raw[label + '_monthly_purchase_amount_' + func + '_max'] = raw[cols].max(axis=1)


# ================================= accumulate function
def accumulate_by_month_lag(df, key, func, label, duration):
    prefix = '_'.join([label, key, func])
    for d in duration:
        col = '_'.join([prefix, 'lag'+str(d)])
        cols = ['_'.join([label, 'month_lag', str(-idx), key, func]) for idx in range(d)]
        df[col] = df[cols].sum(axis=1)
    return df


def ratio_by_month_lag(df, key, func, label, duration):
    prefix = '_'.join([label, key, func])
    loop = duration.copy()
    len_loop = len(loop)
    for i in range(len_loop):
        top = loop.pop(0)
        for bot in loop:
            col_ratio = '_'.join([prefix, 'lag'+str(top), 'to', 'lag'+str(bot)])
            col_top = '_'.join([prefix, 'lag'+str(top)])
            col_bot = '_'.join([prefix, 'lag'+str(bot)])
            df[col_ratio] = df[col_top] / df[col_bot]
    return df


# === purchase_amount - sum & count & mean
print('processing data by month_lag')
lags = [1, 2, 3, 6, 12]
raw = accumulate_by_month_lag(df=raw, key='purchase_amount', func='sum', label='hist', duration=lags)
raw = accumulate_by_month_lag(df=raw, key='purchase_amount', func='count', label='hist', duration=lags)

for l in lags:
    raw['hist_purchase_amount_mean_lag' + str(l)] = \
        raw['hist_purchase_amount_sum_lag' + str(l)] / raw['hist_purchase_amount_count_lag' + str(l)]

raw = ratio_by_month_lag(df=raw, key='purchase_amount', func='sum', label='hist', duration=lags)
raw = ratio_by_month_lag(df=raw, key='purchase_amount', func='count', label='hist', duration=lags)
raw = ratio_by_month_lag(df=raw, key='purchase_amount', func='mean', label='hist', duration=lags)

# === authorized_flag - sum & mean
raw = accumulate_by_month_lag(df=raw, key='authorized_flag', func='sum', label='hist', duration=lags)
raw = accumulate_by_month_lag(df=raw, key='purchase_amount', func='count', label='hist', duration=lags)

for l in lags:
    raw['hist_authorized_flag_mean_lag' + str(l)] = \
        raw['hist_authorized_flag_sum_lag' + str(l)] / raw['hist_purchase_amount_count_lag' + str(l)]

raw = ratio_by_month_lag(df=raw, key='authorized_flag', func='sum', label='hist', duration=lags)
raw = ratio_by_month_lag(df=raw, key='authorized_flag', func='mean', label='hist', duration=lags)

# === remove repeated feature
col_drop = [col for col in raw.columns.values if col.endswith('lag1')]
raw = raw.drop(columns=col_drop)

print('generating div feature')
# === new/hist
for col in raw.columns.values:
    if col.startswith('new_') and raw[col].dtype != object:
        col_bot = 'hist_' + col[4:]
        if col_bot in raw.columns.values:
            col_ratio = col[4:] + '_new_to_hist'
            raw[col_ratio] = raw[col] / raw[col_bot]

# === min/max
for col in raw.columns.values:
    if col.endswith('_min'):
        col_bot = col[:-3] + 'max'
        if col_bot in raw.columns.values:
            col_ratio = col + '_to_max'
            raw[col_ratio] = raw[col] / raw[col_bot]

# === std/mean
for col in raw.columns.values:
    if col.endswith('_mean'):
        col_bot = col[:-4] + 'std'
        if col_bot in raw.columns.values:
            col_ratio = col + '_to_std'
            raw[col_ratio] = raw[col] / raw[col_bot]

# === normalization purchase_amount-related feature
print('normalizing feature')
col_nor = [
    'hist_purchase_amount_per_day',
    'hist_purchase_amount_per_month',
    'hist_monthly_purchase_amount_sum_min',
    'hist_monthly_purchase_amount_sum_max',

    'new_purchase_amount_per_day',
    'new_purchase_amount_per_month',
    'new_monthly_purchase_amount_sum_min',
    'new_monthly_purchase_amount_sum_max',
]
lags.remove(1)
col_nor.extend(['hist_purchase_amount_sum_lag' + str(l) for l in lags])
col_nor.extend(['hist_purchase_amount_mean_lag' + str(l) for l in lags])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_sum')])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_mean')])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_max')])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_min')])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_std')])
col_nor.extend([col for col in raw.columns.values if col.endswith('_purchase_amount_median')])

amount_unit = 0.00001503
raw[col_nor] = (raw[col_nor] - 500) * 100 * amount_unit

# === remove feature with 95% NA value
len_df = len(raw)
col_drop = []
for col in raw.columns.values:
    if raw[col].isnull().sum() / len_df > 0.995:
        col_drop.append(col)
raw = raw.drop(columns=col_drop)

# ================================================================================== Save Data
print('saving file')
raw.to_csv('raw_fe.csv', index=False)
raw.isnull().sum().to_csv('columns_and_null_count.csv')
