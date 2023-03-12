import pandas as pd


df_train = pd.read_csv('train/X_train.csv', index_col=0)
df_test = pd.read_csv('test/X_val.csv', index_col=0)
df_total = pd.concat([df_train, df_test])


cat_columns = []
num_columns = []
for column_name in df_total.columns:
    if (df_total[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]


ohe_cat_col = pd.get_dummies(df_total[cat_columns])
df_total = df_total.join(ohe_cat_col)
df_total.drop(columns=['id', 'code', 'period'], inplace=True)


X_train = df_total.iloc[0:df_train.shape[0],:]
X_test = df_total.iloc[df_train.shape[0]:,:]


X_train.to_csv('/content/X_train.csv', index=False)
X_test.to_csv('/content/X_test.csv', index=False)
