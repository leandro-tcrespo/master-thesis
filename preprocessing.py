import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def preprocess_data(csv_path, cols_to_load, seed):
    data = pd.read_csv(csv_path, header=0)
    X = data[cols_to_load]
    y = data[['diag_multi']]
    features = X.columns
    categorical_features = [name for name in features if name != 'age']
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    datasets = train_test_split(X, y, test_size=0.25, random_state=seed)
    train_data, test_data, train_labels, test_labels = datasets

    if 'age' not in features:
        enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore', ), categorical_features)],
                                sparse_threshold=0, verbose_feature_names_out=False)

    else:
        enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore', ), categorical_features),
                                            ("AgeScaler", MinMaxScaler(), ['age'])],
                                sparse_threshold=0, verbose_feature_names_out=False)

    smote = SMOTENC(random_state=seed, categorical_features=categorical_indices)
    tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
    enn = EditedNearestNeighbours(sampling_strategy='all', n_jobs=-1)
    ros = RandomOverSampler(random_state=seed)

    return train_data, test_data, train_labels, test_labels, enc, ros, tomek, smote, enn
