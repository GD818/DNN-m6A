import scipy.io as sio
import numpy as np
import pandas as pd
from skfeature.function.information_theoretical_based import MRMR
k = 82
name = "H_B"
feature = "DNC"
data_root = r"fused data"
# ________________________________________________________________________________________
result_root = r"mRMR"

# _________________________________________________________________________________________
# result path
train_path = result_root + r'\{0}_{1}_{2}-{3}.csv'.format("mRMR", name, feature, k)
# ______________dataset__________
train_data_path = data_root + r'\{}_{}.csv'.format(name, feature)
df_train = pd.read_csv(train_data_path)
data_train = df_train.iloc[:, 1:].values
label_train = df_train.iloc[:, 0].values
feature_index,_,_ = MRMR.mrmr(data_train, label_train, n_selected_features=k)
transformed_train = data_train[:,feature_index]
print("feature_index", feature_index)
print('transformed_train', transformed_train)
# csv
shu_1 = transformed_train
X_train = pd.DataFrame(data=shu_1)
y_train = pd.DataFrame(label_train, index=range(len(data_train)), columns=['class'])
fill_all = y_train.join([X_train], how='outer')
fill_all.to_csv(train_path, index=0)

