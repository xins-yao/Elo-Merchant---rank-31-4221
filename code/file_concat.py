import pandas as pd
from os import path

# ================================================================================== Parameters
folder = './raw_feature'

# ================================================================================== Single
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
raw = pd.concat([train, test], axis=0, sort=False)

# === hist
files = [
    'raw_single_hist_authorized_flag.csv',

    'raw_single_hist_category_2_1.0.csv',
    'raw_single_hist_category_2_2.0.csv',
    'raw_single_hist_category_2_3.0.csv',
    'raw_single_hist_category_2_4.0.csv',
    'raw_single_hist_category_2_5.0.csv',
    'raw_single_hist_category_2_6.0.csv',
    'raw_single_hist_category_3_A.csv',
    'raw_single_hist_category_3_B.csv',
    'raw_single_hist_category_3_C.csv',
    'raw_single_hist_category_3_D.csv',

    'raw_single_hist_category_1.csv',
    'raw_single_hist_category_2.csv',
    'raw_single_hist_category_4.csv',

    'raw_single_hist_installments.csv',
    'raw_single_hist_purchase_amount.csv',
    'raw_single_hist_purchase_amount_over_max.csv',
    'raw_single_hist_duration.csv',

    'raw_single_hist_year.csv',
    'raw_single_hist_month.csv',
    'raw_single_hist_dayofyear.csv',
    'raw_single_hist_weekofyear.csv',
    'raw_single_hist_dayofweek.csv',
    'raw_single_hist_weekend.csv',
    'raw_single_hist_purchase_date.csv',
    'raw_single_hist_month_lag.csv',
    'raw_single_hist_month_diff.csv',
    'raw_single_hist_purchase_date_to_ref.csv',

    'raw_single_hist_merchant_id.csv',
    'raw_single_hist_merchant_category_id.csv',
    'raw_single_hist_state_id.csv',
    'raw_single_hist_city_id.csv',
    'raw_single_hist_subsector_id.csv',
    'raw_single_hist_merchant_group_id.csv',

    'raw_single_hist_Christmas_Day_2017.csv',
    'raw_single_hist_Mothers_Day_2017.csv',
    'raw_single_hist_Fathers_day_2017.csv',
    'raw_single_hist_Children_day_2017.csv',
    'raw_single_hist_Valentine_Day_2017.csv',
    'raw_single_hist_Black_Friday_2017.csv',
    'raw_single_hist_Mothers_Day_2018.csv',

    'raw_single_hist_purchase_price.csv'
]

for f in files:
    print('Merging {}'.format(f))
    df = pd.read_csv(path.join(folder, f))
    raw = pd.merge(raw, df, how='left', on='card_id')
    del df

# === new
files = [
    'raw_single_new_category_2_1.0.csv',
    'raw_single_new_category_2_2.0.csv',
    'raw_single_new_category_2_3.0.csv',
    'raw_single_new_category_2_4.0.csv',
    'raw_single_new_category_2_5.0.csv',
    'raw_single_new_category_2_6.0.csv',
    'raw_single_new_category_3_A.csv',
    'raw_single_new_category_3_B.csv',
    'raw_single_new_category_3_C.csv',
    'raw_single_new_category_3_D.csv',

    'raw_single_new_category_1.csv',
    'raw_single_new_category_2.csv',
    'raw_single_new_category_4.csv',

    'raw_single_new_installments.csv',
    'raw_single_new_purchase_amount.csv',
    'raw_single_new_purchase_amount_over_max.csv',
    'raw_single_new_duration.csv',

    'raw_single_new_year.csv',
    'raw_single_new_month.csv',
    'raw_single_new_weekofyear.csv',
    'raw_single_new_dayofweek.csv',
    'raw_single_new_weekend.csv',
    'raw_single_new_purchase_date.csv',
    'raw_single_new_month_lag.csv',
    'raw_single_new_month_diff.csv',
    'raw_single_new_purchase_date_to_ref.csv',

    'raw_single_new_merchant_id.csv',
    'raw_single_new_merchant_category_id.csv',
    'raw_single_new_state_id.csv',
    'raw_single_new_city_id.csv',
    'raw_single_new_subsector_id.csv',
    'raw_single_new_merchant_group_id.csv',

    'raw_single_new_Christmas_Day_2017.csv',
    'raw_single_new_Children_day_2017.csv',
    'raw_single_new_Black_Friday_2017.csv',
    'raw_single_new_Mothers_Day_2018.csv',

    'raw_single_new_purchase_price.csv'
]

for f in files:
    print('Merging {}'.format(f))
    df = pd.read_csv(path.join(folder, f))
    raw = pd.merge(raw, df, how='left', on='card_id')
    del df

# ================================================================================== Multi - new
# === installments
print('Merging raw_multi_new_installments.csv')
df = pd.read_csv(path.join(folder, 'raw_multi_new_installments.csv'))
col_drop = [
    'new_installments_12.0_purchase_amount_std',
    'new_installments_6.0_purchase_amount_std',
    'new_installments_4.0_purchase_amount_std',
    'new_installments_10.0_purchase_amount_std',
    'new_installments_5.0_purchase_amount_std',
    'new_installments_9.0_purchase_amount_min',
    'new_installments_9.0_purchase_amount_max',
    'new_installments_9.0_purchase_amount_sum',
    'new_installments_9.0_purchase_amount_mean',
    'new_installments_9.0_purchase_amount_std',
    'new_installments_9.0_purchase_amount_count',
    'new_installments_8.0_purchase_amount_min',
    'new_installments_8.0_purchase_amount_max',
    'new_installments_8.0_purchase_amount_sum',
    'new_installments_8.0_purchase_amount_mean',
    'new_installments_8.0_purchase_amount_std',
    'new_installments_8.0_purchase_amount_count',
    'new_installments_7.0_purchase_amount_min',
    'new_installments_7.0_purchase_amount_max',
    'new_installments_7.0_purchase_amount_sum',
    'new_installments_7.0_purchase_amount_mean',
    'new_installments_7.0_purchase_amount_std',
    'new_installments_7.0_purchase_amount_count',
    'new_installments_11.0_purchase_amount_min',
    'new_installments_11.0_purchase_amount_max',
    'new_installments_11.0_purchase_amount_sum',
    'new_installments_11.0_purchase_amount_mean',
    'new_installments_11.0_purchase_amount_std',
    'new_installments_11.0_purchase_amount_count',
]
df = df.drop(columns=col_drop)
raw = pd.merge(raw, df, how='left', on='card_id')
del df

# === category_3
print('Merging raw_multi_new_category_3.csv')
df = pd.read_csv(path.join(folder, 'raw_multi_new_category_3.csv'))
cols = [col for col in df.columns.values if 'category_3_C' in col]
cols.append('card_id')
raw = pd.merge(raw, df[cols], how='left', on='card_id')
del df

# === others
files = [
    'raw_multi_new_year.csv',
    'raw_multi_new_weekend.csv',
    'raw_multi_new_month_lag.csv',
    'raw_multi_new_month.csv',
    'raw_multi_new_dayofweek.csv',
    'raw_multi_new_category_2.csv',
    'raw_multi_new_category_1.csv'
]

for f in files:
    print('Merging {}'.format(f))
    df = pd.read_csv(path.join(folder, f))
    raw = pd.merge(raw, df, how='left', on='card_id')
    del df

# ================================================================================== Multi - hist
# === installments
print('Merging raw_multi_hist_installments.csv')
df = pd.read_csv(path.join(folder, 'raw_multi_hist_installments.csv'))
col_drop = [
    'hist_installments_7.0_purchase_amount_std',
    'hist_installments_9.0_purchase_amount_std',
    'hist_installments_11.0_authorized_flag_mean',
    'hist_installments_11.0_authorized_flag_sum',
    'hist_installments_11.0_purchase_amount_min',
    'hist_installments_11.0_purchase_amount_max',
    'hist_installments_11.0_purchase_amount_sum',
    'hist_installments_11.0_purchase_amount_mean',
    'hist_installments_11.0_purchase_amount_std',
    'hist_installments_11.0_purchase_amount_count',
]
df = df.drop(columns=col_drop)
raw = pd.merge(raw, df, how='left', on='card_id')
del df

# === category_3
print('Merging raw_multi_hist_category_3.csv')
df = pd.read_csv(path.join(folder, 'raw_multi_hist_category_3.csv'))
cols = [col for col in df.columns.values if 'category_3_C' in col]
cols.append('card_id')
raw = pd.merge(raw, df[cols], how='left', on='card_id')
del df

# === others
files = [
    'raw_multi_hist_year.csv',
    'raw_multi_hist_weekend.csv',
    'raw_multi_hist_month_lag.csv',
    'raw_multi_hist_month.csv',
    'raw_multi_hist_dayofweek.csv',
    'raw_multi_hist_category_2.csv',
    'raw_multi_hist_category_1.csv',
]

for f in files:
    print('Merging {}'.format(f))
    df = pd.read_csv(path.join(folder, f))
    raw = pd.merge(raw, df, how='left', on='card_id')
    del df

# ================================================================================== Save File
print('Saving File')
raw.to_csv(path.join(folder, 'raw_concat.csv'), index=False)
