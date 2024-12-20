import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(csv_path):
    data = pd.read_csv(csv_path, header=0)
    X = data[['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']]
    y = data[['diag_multi']]
    features = X.columns
    categorical_features = [name for name in features if name != 'age']

    datasets = train_test_split(X, y, test_size=0.25, random_state=42)
    train_data, test_data, train_labels, test_labels = datasets

    enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore', ), categorical_features)],
                            remainder='passthrough', sparse_threshold=0, verbose_feature_names_out=False)

    smote_os = SMOTENC(random_state=42, k_neighbors=1, categorical_features=categorical_features)
    random_os = RandomOverSampler(random_state=42)

    return train_data, test_data, train_labels, test_labels, enc, smote_os, random_os
