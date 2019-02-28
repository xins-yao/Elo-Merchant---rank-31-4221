import sys
import numpy as np
import pandas as pd

from time import time


# === reduce memory usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        # print('Processing Column: {}'.format(col))
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_max < np.iinfo(np.int8).max and c_min > np.iinfo(np.int8).min:
                    df[col] = df[col].astype(np.int8)
                elif  c_max < np.iinfo(np.int16).max and c_min > np.iinfo(np.int16).min:
                    df[col] = df[col].astype(np.int16)
                elif c_max < np.iinfo(np.int32).max and c_min > np.iinfo(np.int32).min:
                    df[col] = df[col].astype(np.int32)
                # elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    # df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_max < np.finfo(np.float16).max and c_min > np.finfo(np.float16).min:
                    df[col] = df[col].astype(np.float16)
                elif c_max < np.finfo(np.float32).max and c_min > np.finfo(np.float32).min:
                    df[col] = df[col].astype(np.float32)
                # else:
                    # df[col] = df[col].astype(np.float64)
        elif col == 'card_id':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# === fill NA value in columns with keyword
def fill_na(df, value, keywords: list):
    cols = [col for col in df.columns.values
            if (np.mean([int(word in col) for word in keywords]) == 1)]
    df[cols] = df[cols].fillna(value)
    return df


# === aggregate by single key
def single_agg(args):
    df = args[0]
    key = args[1]
    agg_func = args[2]
    df_merge = args[3]
    label = args[4]

    for k, v in agg_func.items():
        st_time = time()
        print('aggregate ' + label + ' by: ' + str(k))
        agg_df = df[[key, k]].groupby(key)[k].agg(v)
        agg_df.columns = [(label + '_' + k + '_' + col) for col in agg_df.columns.values]
        agg_df.reset_index(inplace=True)
        df_merge = pd.merge(df_merge, agg_df, how='left', on=key)
        print('elapsed time: ', time()-st_time)
        df_merge.to_csv('raw_single_' + label + '_' + k + '.csv', index=False)
    # df_merge.to_csv('raw_single_' + label + '.csv', index=False)
    return df_merge


# === aggregate by multiple key
def multi_level_agg(args):
    df = args[0]
    key = args[1]
    level = args[2]
    agg_func = args[3]
    df_merge = args[4]
    label = args[5]

    print('processing multi-level aggregation by level ' + level)
    agg_df = df.groupby([key, level]).agg(agg_func)
    level_set = df[level].unique()
    for l in level_set:
        # if np.isnan(l):
            # break
        st_time = time()
        df_slice = agg_df.xs(key=l, level=level)
        prefix = '_'.join([label, level, str(l)])
        df_slice.columns = [prefix + '_' + '_'.join(col) for col in agg_df.columns.values]
        df_merge = pd.merge(df_merge, df_slice.reset_index(), how='left', on=key)
        print('label: {}, level: {} {}, elapsed time: {}'.format(label, level, l, time() - st_time))

    df_merge.to_csv('raw_multi_' + label + '_' + level + '.csv', index=False)
    return df_merge


# === aggregation func
def repurchase_max(x):
    return x.value_counts().max()


def repurchase_mean(x):
    return len(x) / x.nunique()


def repurchase_count(x):
    return len(x) - x.nunique()


def entropy(x):
    prob = x.value_counts() / len(x)
    return np.sum(-1 * prob * np.log2(prob))


def mode(x):
    return x.mode()[0]


def diff_max(x):
    diff = np.diff(x)
    if len(diff) > 0:
        return diff.max()
    else:
        return np.nan


def diff_min(x):
    diff = np.diff(x)
    if len(diff) > 0:
        return diff.min()
    else:
        return np.nan


# === Uniform Distribution
def uni_distribution(df, key):
    df['rounded_target'] = df['target'].round(0)
    df = df.sort_values('rounded_target').reset_index(drop=True)
    vc = df['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df1 = pd.DataFrame()    # TODO: what's df1 & df2 ??????????
    df['indexcol'], i = 0, 1
    for k, v in vc.items():
        step = df.shape[0] / v
        indent = df.shape[0] / (v + 1)
        df2 = df[df['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j * step + 0.000001 * i
        df1 = pd.concat([df2, df1])
        i += 1
    df = df1.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del df['indexcol']
    del df['rounded_target']
    return df


# === logging
class Logger(object):
    def __init__(self, file="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def is_in_holiday(x):
    return x if 0 < x < 100 else 0
