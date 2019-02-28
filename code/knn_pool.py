import time
import numpy as np
import pandas as pd

from os import path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer


def knn_mean(args):
    df = args[0]
    mmin = args[1]
    mmax = args[2]
    print('processing from {} to {}'.format(mmin, mmax))
    for i in range(mmin, mmax):
        idx = list(df.loc[i, 'nn_500'])
        if i in idx:
            idx.remove(i)
        df.loc[i, 'nn_500_mean_target'] = df.loc[idx, 'target'].mean()
    print('saving data')
    df = df[['card_id', 'nn_500_mean_target']]
    print('debug')
    df.to_csv('./raw_feature/raw_date_knn_500_' + str(mmin) + '-' + str(mmax) + '.csv', index=False)
    print('done')


# ================================================================
if __name__ == '__main__':
    len_train = 201917
    len_test = 123623
    min_target = -33.21928095

    top_folder = './CV_LB_log'
    feats_folder = './raw_feature/selector-null importance-v4'
    data_folder = './raw_feature'

    # ================================================================
    print('load data')
    # train = pd.read_csv('./input/train.csv')
    # test = pd.read_csv('./input/test.csv')

    feats = [
        'hist_purchase_date_max',
        'new_purchase_date_max',
        'purchase_date_max_new_to_hist',
        'new_purchase_date_min',
        'new_purchase_date_to_ref_max',
        'hist_month_diff_min',
    ]
    col_read = feats.copy()
    col_read.extend(['card_id', 'target'])
    raw = pd.read_csv(path.join(data_folder, 'raw_fe-v20-add holiday feature.csv'), usecols=col_read)
    raw = raw.replace(np.inf, np.nan)
    raw = raw.replace(-np.inf, np.nan)
    raw = raw.fillna(0)

    # === KNN
    train = raw[:len_train]
    x_train = train[feats]

    print('knn')
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=500, n_jobs=12)
    nbrs.fit(x_train)
    nn = nbrs.kneighbors(raw[feats], return_distance=False)
    elapsed_time = (time.time() - start_time) / 60
    print('elapsed_time: {} minutes'.format(elapsed_time))

    raw['nn_500'] = list(nn)
    raw['nn_500_mean_target'] = None

    from multiprocessing import Pool, cpu_count
    n_thread = cpu_count()
    pool = Pool(6)

    argss = [
        [raw, 0, 30000],
        [raw, 30000, 60000],
        [raw, 60000, 90000],
        [raw, 90000, 120000],
        [raw, 120000, 150000],
        [raw, 150000, 180000],
        [raw, 180000, 210000],
        [raw, 210000, 240000],
        [raw, 240000, 270000],
        [raw, 270000, 300000],
        [raw, 300000, 325540]
    ]

    pool.map(knn_mean, argss)
    pool.close()
    pool.join()


