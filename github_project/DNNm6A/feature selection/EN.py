import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from main_file import elasticNet

# ________________________________________________________________________________________
k = 0.09  # parameter value
A = "09"

name = "data"
feature = "binary"
data_root = r"examples"
# ________________________________________________________________________________________


# _________________________________________________________________________________________
# result root
train_path = r'..\{0}_{1}_{2}-{3}.csv'.format("EN", name, feature, A)
# ______________train_data__________________________
train_data_path = data_root + r'\binary.csv'
df_train = pd.read_csv(train_data_path)
data_train = df_train.iloc[:, 1:].values
label_train = df_train.iloc[:, 0].values
sc = StandardScaler().fit(data_train)
shu_train = sc.transform(data_train)

index, data_train_index = \
    elasticNet(shu_train, label_train, k)
# csv
y_train = pd.DataFrame(label_train, index=range(len(data_train)), columns=['class'])
fill_all = y_train.join([data_train_index], how='outer')
fill_all.to_csv(train_path, index=0)


