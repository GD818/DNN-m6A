import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from main_file import SVD

result_root = r"SVD"
data_root = r"fused data"
# ________________________________________________________________________________________
K = 82
name = "H_B"
feature = "DNC"
# ________________________________________________________________________________________
# # result path
train_path = result_root + r'\{0}_{1}_{2}-{3}.csv'.format("TSVD", name, feature, K)
# ______________dataset__________________________
train_data_path = data_root + r'\{}_{}.csv'.format(name, feature)
df_train = pd.read_csv(train_data_path)
data_train = df_train.iloc[:, 1:].values
label_train = df_train.iloc[:, 0].values
sc = StandardScaler().fit(data_train)
shu_train = sc.transform(data_train)
# ———————————————result—————————————————————
new_data = SVD(shu_train, K)
# csv
y_train = pd.DataFrame(label_train, index=range(len(data_train)), columns=['class'])
fill_all = y_train.join([new_data], how='outer')
fill_all.to_csv(train_path, index=0)
