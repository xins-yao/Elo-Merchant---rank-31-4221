import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
hist_trans = pd.read_csv('./input/historical_transactions.csv', usecols=['card_id', 'merchant_id'])
bad_merchant = pd.read_csv('./raw_feature/bad_merchant_id.csv', usecols=['merchant_id'])

hist_trans = hist_trans.dropna()

# === count vector
count_vec = CountVectorizer()
count_vec.fit(bad_merchant['merchant_id'])

df_agg = hist_trans.groupby('card_id')['merchant_id'].agg(['unique'])
df_agg['merchant_list'] = df_agg['unique'].apply(lambda x: ' '.join(x))
df_agg = df_agg.reset_index()

vector = count_vec.transform(df_agg['merchant_list']).toarray()
vector_train = count_vec.transform(df_agg.loc[df_agg['card_id'].isin(train['card_id']), 'merchant_list']).toarray()

# === KMeans
kmeans = KMeans(n_clusters=12, random_state=2019, n_jobs=12)
kmeans.fit(vector_train)
label = kmeans.predict(vector)
df_agg['cluster'] = label

raw = pd.concat([train, test], axis=0, sort=False)
raw = pd.merge(raw, df_agg[['card_id', 'cluster']], how='left', on='card_id')

df_mean_target = raw.groupby('cluster')['target'].agg(['mean'])
df_mean_target.columns = ['cluster_mean_target']
df_mean_target = df_mean_target.reset_index()
raw = pd.merge(raw, df_mean_target[['cluster', 'cluster_mean_target']], how='left', on='cluster')

# === Save
raw = raw[['card_id', 'cluster', 'cluster_mean_target']]
raw.to_csv('./raw_feature/raw_kmeans.csv', index=False)


