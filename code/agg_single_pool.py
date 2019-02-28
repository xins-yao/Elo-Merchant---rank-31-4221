import gc
import datetime
import warnings
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from utils import single_agg, reduce_mem_usage
from utils import entropy, repurchase_max, repurchase_mean, repurchase_count, mode
from utils import diff_max, diff_min
from utils import is_in_holiday


# ================================================================================== Main
if __name__ == '__main__':

    n_thread = cpu_count()
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ================================================================================== Load Data
    hist_trans = pd.read_csv('./input/historical_transactions.csv', parse_dates=['purchase_date'],
                             usecols=['card_id', 'purchase_date', 'month_lag'])
    new_trans = pd.read_csv('./input/new_merchant_transactions.csv', parse_dates=['purchase_date'],
                            usecols=['card_id', 'purchase_date', 'month_lag'])
    merchant = pd.read_csv('./input/merchants.csv')
    train = pd.read_csv('./input/train.csv', parse_dates=['first_active_month'])
    test = pd.read_csv('./input/test.csv', parse_dates=['first_active_month'])
    raw = pd.concat([train, test], axis=0, sort=False)
    print('Load Data Done.')

    # ================================================================================== Pre-processing
    ref_date = datetime.date(2018, 5, 1)
    # max_amount = 2.259
    # amount_unit = 0.00001503

    # hist_trans = pd.merge(hist_trans, merchant[['merchant_id', 'merchant_group_id', 'category_4']], how='left',
    #                       on='merchant_id')
    # new_trans = pd.merge(new_trans, merchant[['merchant_id', 'merchant_group_id', 'category_4']], how='left',
    #                      on='merchant_id')

    # holiday - hist
    # hist_trans['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # hist_trans['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # hist_trans['Fathers_day_2017'] = (pd.to_datetime('2017-08-13') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # hist_trans['Children_day_2017'] = (pd.to_datetime('2017-10-12') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # hist_trans['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # hist_trans['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - hist_trans['purchase_date']).dt.days.apply(is_in_holiday)

    # holiday - new
    # new_trans['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # new_trans['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # new_trans['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_trans['purchase_date']).dt.days.apply(is_in_holiday)
    # new_trans['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - new_trans['purchase_date']).dt.days.apply(is_in_holiday)

    # hist_trans['ref_date'] = hist_trans.apply(lambda x: x['purchase_date'] - pd.DateOffset(months=x['month_lag']), axis=1)
    # new_trans['ref_date'] = new_trans.apply(lambda x: x['purchase_date'] - pd.DateOffset(months=x['month_lag'] - 1), axis=1)

    hist_trans = pd.merge(hist_trans, raw[['card_id', 'first_active_month']], how='left', on='card_id')
    new_trans = pd.merge(new_trans, raw[['card_id', 'first_active_month']], how='left', on='card_id')

    for df in [hist_trans, new_trans]:
        # missing value
        # df['category_3'].fillna('D', inplace=True)
        # df['category_2'].fillna(6.0, inplace=True)
        # df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        # df['installments'] = df['installments'].replace(999, np.nan)
        # df['installments'] = df['installments'].replace(-1, np.nan)

        # reverse normalization
        # df['purchase_amount_over_max'] = (df['purchase_amount'] > max_amount)
        # df['purchase_amount'] = np.clip(df['purchase_amount'], a_min=-1, a_max=max_amount)
        # df['purchase_amount'] = df['purchase_amount']/amount_unit/100 + 500

        # df_ins = df['installments'].replace(0, 1)
        # df['purchase_price'] = df['purchase_amount'] / df_ins

        # time-related
        df['year'] = df['purchase_date'].dt.year
        df['month'] = df['purchase_date'].dt.month
        # df['dayofyear'] = df['purchase_date'].dt.dayofyear
        # df['weekofyear'] = df['purchase_date'].dt.weekofyear
        # df['dayofweek'] = df['purchase_date'].dt.dayofweek
        # df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)

        df['purchase_date_to_active'] = (df['purchase_date'].dt.date - df['first_active_month'].dt.date).dt.days

        # ref_date per card
        ref_year = df['year'] + (df['month'] - df['month_lag'] - 0.1) // 12
        ref_month = (df['month'] - df['month_lag']) % 12
        ref_month = ref_month.replace(0, 12)
        df['ref_date'] = ref_year.astype(int).astype(str) + '-' + ref_month.astype(str)
        df['ref_date'] = pd.to_datetime(df['ref_date'])

        df['purchase_date_to_ref'] = (df['purchase_date'].dt.date - df['ref_date'].dt.date).dt.days
        df['purchase_date'] = (df['purchase_date'].dt.date - ref_date).dt.days
        # df['month_diff'] = (ref_date - df['ref_date'].dt.date).dt.days // 30

        # category
        # df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        # df['category_4'] = df['category_4'].map({'Y': 1, 'N': 0})
        # df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})

        # duration
        # df['duration'] = df['purchase_amount'] * df['month_diff']

        # del df['ref_date']
        del df['first_active_month']

    # ================================================================================== Reduce Memory Usage
    hist_trans = reduce_mem_usage(hist_trans)
    new_trans = reduce_mem_usage(new_trans)

    # uauth_trans = hist_trans[hist_trans['authorized_flag'] == 0]
    #     # del hist_trans
    #     # del new_trans

    # ================================================================================== Single Aggregation
    # df_category = hist_trans['category_2']
    # hist_trans = pd.get_dummies(hist_trans, prefix=['category_2', 'category_3'], columns=['category_2', 'category_3'])
    # hist_trans = hist_trans.join(df_category)

    # df_category = new_trans['category_2']
    # new_trans = pd.get_dummies(new_trans, prefix=['category_2', 'category_3'], columns=['category_2', 'category_3'])
    # new_trans = new_trans.join(df_category)

    # df_category = uauth_trans['category_2']
    # uauth_trans = pd.get_dummies(uauth_trans, prefix=['category_2', 'category_3'], columns=['category_2', 'category_3'])
    # uauth_trans = uauth_trans.join(df_category)

    agg_hist = {
        # 'authorized_flag': ['mean', 'sum'],

        # 'category_2_1.0': ['mean'],
        # 'category_2_2.0': ['mean'],
        # 'category_2_3.0': ['mean'],
        # 'category_2_4.0': ['mean'],
        # 'category_2_5.0': ['mean'],
        # 'category_2_6.0': ['mean'],
        # 'category_3_A': ['mean'],
        # 'category_3_B': ['mean'],
        # 'category_3_C': ['mean'],
        # 'category_3_D': ['mean'],
        #
        # 'category_1': ['mean'],
        # 'category_2': ['nunique', entropy, mode],
        # 'category_4': ['mean'],

        # 'installments': ['sum', 'mean', 'max', 'min', 'std', 'nunique'],
        # 'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'median', 'count'],
        # 'purchase_amount_over_max': ['sum', 'mean'],
        # 'duration': ['sum', 'mean', 'max', 'min', 'std', 'median'],
        # 'purchase_price': ['sum', 'mean', 'max', 'min', 'std', 'median'],

        # 'year': ['nunique', 'std'],
        # 'month': ['nunique', 'std', entropy],
        # 'dayofyear': ['nunique', 'std', entropy],
        # 'weekofyear': ['nunique', 'std', entropy],
        # 'dayofweek': ['nunique', 'std', entropy],
        # 'weekend': ['mean', 'std'],
        # 'purchase_date': ['nunique', 'std', 'max', 'min', np.ptp, entropy, diff_max, diff_min],
        # 'month_lag': ['nunique', 'std', 'mean', 'max', 'min', entropy],
        # 'month_diff': ['min'],
        # 'purchase_date_to_ref': ['std', 'max', 'min', entropy],
        'purchase_date_to_active': ['max', 'skew'],
        'purchase_date_to_ref': ['skew'],
        'purchase_date': ['skew'],

        # 'merchant_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_category_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'state_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'city_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'subsector_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_group_id': ['nunique', repurchase_max, repurchase_count, entropy]

        # 'Christmas_Day_2017': ['mean'],
        # 'Mothers_Day_2017': ['mean'],
        # 'Fathers_day_2017': ['mean'],
        # 'Children_day_2017': ['mean'],
        # 'Valentine_Day_2017': ['mean'],
        # 'Black_Friday_2017': ['mean'],
        # 'Mothers_Day_2018': ['mean'],
    }

    agg_new = {
        # 'category_2_1.0': ['mean'],
        # 'category_2_2.0': ['mean'],
        # 'category_2_3.0': ['mean'],
        # 'category_2_4.0': ['mean'],
        # 'category_2_5.0': ['mean'],
        # 'category_2_6.0': ['mean'],
        # 'category_3_A': ['mean'],
        # 'category_3_B': ['mean'],
        # 'category_3_C': ['mean'],
        # 'category_3_D': ['mean'],
        #
        # 'category_1': ['mean'],
        # 'category_2': ['nunique', entropy, mode],
        # 'category_4': ['mean'],

        # 'installments': ['sum', 'mean', 'max', 'min', 'std', 'nunique'],
        # 'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'median', 'count'],
        # 'purchase_amount_over_max': ['sum', 'mean'],
        # 'duration': ['sum', 'mean', 'max', 'min', 'std', 'median'],
        # 'purchase_price': ['sum', 'mean', 'max', 'min', 'std', 'median'],

        # 'year': ['nunique', 'std'],
        # 'month': ['nunique', 'std', entropy],
        # 'weekofyear': ['nunique', 'std', entropy],
        # 'dayofweek': ['nunique', 'std', entropy],
        # 'weekend': ['mean', 'std'],
        # 'purchase_date': ['nunique', 'std', 'max', 'min', np.ptp, entropy, diff_max, diff_min],
        # 'month_lag': ['nunique', 'std', 'mean', 'max', 'min', entropy],
        # 'month_diff': ['min'],
        # 'purchase_date_to_ref': ['std', 'max', 'min', entropy],
        'purchase_date_to_active': ['max', 'skew'],
        'purchase_date_to_ref': ['skew'],
        'purchase_date': ['skew'],

        # 'merchant_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_category_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'state_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'city_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'subsector_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_group_id': ['nunique', repurchase_max, repurchase_count, entropy]

        # 'Christmas_Day_2017': ['mean'],
        # 'Children_day_2017': ['mean'],
        # 'Black_Friday_2017': ['mean'],
        # 'Mothers_Day_2018': ['mean'],
    }

    agg_uauth = {
        # 'category_2_1.0': ['mean'],
        # 'category_2_2.0': ['mean'],
        # 'category_2_3.0': ['mean'],
        # 'category_2_4.0': ['mean'],
        # 'category_2_5.0': ['mean'],
        # 'category_2_6.0': ['mean'],
        # 'category_3_A': ['mean'],
        # 'category_3_B': ['mean'],
        # 'category_3_C': ['mean'],
        # 'category_3_D': ['mean'],

        # 'category_1': ['mean'],
        # 'category_2': ['nunique', entropy, mode],
        # 'category_4': ['mean'],

        # 'installments': ['sum', 'mean', 'max', 'min', 'std', 'nunique'],
        # 'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'median', 'count'],
        # 'purchase_amount_over_max': ['sum', 'mean'],
        # 'duration': ['sum', 'mean', 'max', 'min', 'std', 'median'],
        # 'purchase_price': ['sum', 'mean', 'max', 'min', 'std', 'median'],

        # 'year': ['nunique', 'std'],
        # 'month': ['nunique', 'std', entropy],
        # 'weekofyear': ['nunique', 'std', entropy],
        # 'dayofweek': ['nunique', 'std', entropy],
        # 'weekend': ['mean', 'std'],
        # 'purchase_date': ['nunique', 'std', 'max', 'min', np.ptp, entropy, diff_max, diff_min],
        # 'month_lag': ['nunique', 'std', 'mean', 'max', 'min', entropy],
        # 'month_diff': ['min', 'nunique'],
        # 'purchase_date_to_ref': ['std', 'max', 'min', entropy],

        # 'merchant_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_category_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'state_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'city_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'subsector_id': ['nunique', repurchase_max, repurchase_mean, repurchase_count, entropy],
        # 'merchant_group_id': ['nunique', repurchase_max, repurchase_count, entropy]

        # 'Christmas_Day_2017': ['mean'],
        # 'Children_day_2017': ['mean'],
        # 'Black_Friday_2017': ['mean'],
        # 'Mothers_Day_2018': ['mean'],
    }

    df_merge = pd.DataFrame(raw['card_id'])

    # === careful for MEMORY ERROR
    pool = Pool(n_thread)
    argss_hist = [[hist_trans, 'card_id', {k: v}, df_merge, 'hist'] for k, v in agg_hist.items()]
    argss_new = [[new_trans, 'card_id', {k: v}, df_merge, 'new'] for k, v in agg_new.items()]
    argss = argss_hist + argss_new

    # argss_uauth = [[uauth_trans, 'card_id', {k: v}, df_merge, 'uauth'] for k, v in agg_uauth.items()]
    # argss = argss_hist + argss_new + argss_uauth

    pool.map(single_agg, argss)

    pool.close()
    pool.join()


