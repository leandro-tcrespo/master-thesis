import pandas as pd
from sklearn.model_selection import train_test_split

import feature_loader
import hkb

data = pd.read_csv("../data/data.csv", header=0)
X = data[feature_loader.no_lab_features]
y = data[['diag_multi']]
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=1040605528)

hkb.fit(train_data, train_labels, "", "2}Alter_jung,Alter_mittel,Alter_alt",
        f"hkb_temp.kb",f"hkb_train_data_temp.txt", train_data.shape[1])
