import datetime
import warnings
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from utils import multi_level_agg, reduce_mem_usage


# ================================================================================== Main
if __name__ == '__main__':

    n_thread = cpu_count()
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ================================================================================== Load Data
    hist_trans = pd.read_csv('./input/historical_transactions.csv', parse_dates=['purchase_date'],
                             usecols=['card_id', 'authorized_flag', 'purchase_date', 'merchant_id', 'category_1'])
    # new_trans = pd.read_csv('./input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
    merchant = pd.read_csv('./input/merchants.csv')
    train = pd.read_csv('./input/train.csv', parse_dates=['first_active_month'])
    test = pd.read_csv('./input/test.csv', parse_dates=['first_active_month'])
    raw = pd.concat([train, test], axis=0, sort=False)
    print('Load Data Done.')
    # ================================================================================== Pre-processing
    ref_date = datetime.date(2018, 5, 1)
    max_amount = 2.259
    amount_unit = 0.00001503

    # hist_trans = pd.merge(hist_trans, merchant[['merchant_id', 'merchant_group_id', 'category_4']], how='left',
    #                       on='merchant_id')
    # new_trans = pd.merge(new_trans, merchant[['merchant_id', 'merchant_group_id', 'category_4']], how='left',
    #                      on='merchant_id')

    # for df in [hist_trans, new_trans]:
    for df in [hist_trans]:
        # missing value
        # df['category_3'].fillna('D', inplace=True)
        # df['category_2'].fillna(6.0, inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        # df['installments'] = df['installments'].replace(999, np.nan)
        # df['installments'] = df['installments'].replace(-1, np.nan)

        # reverse normalization
        # df['purchase_amount'] = np.clip(df['purchase_amount'], a_min=-1, a_max=max_amount)
        # df['purchase_amount'] = df['purchase_amount']/amount_unit/100 + 500

        # time-related
        # df['year'] = df['purchase_date'].dt.year
        # df['month'] = df['purchase_date'].dt.month
        # df['dayofyear'] = df['purchase_date'].dt.dayofyear
        # df['weekofyear'] = df['purchase_date'].dt.weekofyear
        # df['dayofweek'] = df['purchase_date'].dt.dayofweek
        # df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)

        # category
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        # df['category_4'] = df['category_4'].map({'Y': 1, 'N': 0})
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})

    # ================================================================================== Reduce Memory Usage
    hist_trans = reduce_mem_usage(hist_trans)
    # new_trans = reduce_mem_usage(new_trans)

    uauth_trans = hist_trans[hist_trans['authorized_flag'] == 0]
    # del hist_trans
    # del new_trans

    # ================================================================================== Multi-level Aggregation
    agg_hist = {
        'authorized_flag': ['mean', 'sum'],
        'purchase_amount': ['min', 'max', 'sum', 'mean', 'std', 'count', 'median']
    }

    agg_new = {'purchase_amount': ['min', 'max', 'sum', 'mean', 'std', 'count', 'median']}

    df_merge = pd.DataFrame(raw['card_id'])

    agg_merchant = {'merchant_id': ['nunique', 'count']}

    # === careful for MEMORY ERROR !!!
    pool = Pool(n_thread)
    argss = [
        [hist_trans, 'card_id', 'category_1', agg_merchant, df_merge, 'hist'],
        [uauth_trans, 'card_id', 'category_1', agg_merchant, df_merge, 'uauth'],

        # [hist_trans, 'card_id', 'installments', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'month_lag', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'category_1', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'category_2', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'category_3', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'year', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'month', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'dayofweek', agg_hist, df_merge, 'hist'],
        # [hist_trans, 'card_id', 'weekend', agg_hist, df_merge, 'hist'],

        # [new_trans, 'card_id', 'installments', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'month_lag', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'category_1', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'category_2', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'category_3', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'year', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'month', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'dayofweek', agg_new, df_merge, 'new'],
        # [new_trans, 'card_id', 'weekend', agg_new, df_merge, 'new'],

        # [uauth_trans, 'card_id', 'installments', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'month_lag', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'category_1', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'category_2', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'category_3', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'year', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'month', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'dayofweek', agg_new, df_merge, 'uauth'],
        # [uauth_trans, 'card_id', 'weekend', agg_new, df_merge, 'uauth'],
    ]
    pool.map(multi_level_agg, argss)

    pool.close()
    pool.join()

