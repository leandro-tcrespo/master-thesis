import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def preprocess_data(csv_path):
    # for reproducibility of the train test split
    seed = np.random.randint(0, 2**31-1,)
    data = pd.read_csv(csv_path, header=0)
    X = data[['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']]
    y = data[['diag_multi']]
    features = X.columns
    categorical_features = [name for name in features if name != 'age']
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    datasets = train_test_split(X, y, test_size=0.25, random_state=seed)
    train_data, test_data, train_labels, test_labels = datasets

    enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore', ), categorical_features),
                                        ("AgeScaler", MinMaxScaler(), ['age'])],
                            sparse_threshold=0, verbose_feature_names_out=False)

    dt_enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore', ), categorical_features)],
                            sparse_threshold=0, verbose_feature_names_out=False, remainder='passthrough')

    smote_os = SMOTENC(categorical_features=categorical_indices)
    smote_enn = SMOTEENN(smote=clone(smote_os))
    smote_tomek = SMOTETomek(smote=clone(smote_os))
    ros = RandomOverSampler()
    tomek = TomekLinks(sampling_strategy='all')

    return (train_data, test_data, train_labels, test_labels, enc, dt_enc,
            smote_enn, smote_tomek, ros, tomek, smote_os, seed)
