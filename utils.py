import pandas as pd
from sklearn import clone

def resample_data(enc, train_d, train_l, os=None, us=None):
    train_d_res = train_d.copy()
    train_l_res = train_l.copy()
    columns = train_d_res.columns
    if os is not None:
        os = clone(os)
        train_d_res, train_l_res = os.fit_resample(train_d, train_l)
    if us is not None:
        enc = clone(enc)
        us = clone(us)
        if 'age' in columns:
            cat_names = [name for name in train_d if name != 'age']
            age_index = train_d_res.columns.get_loc("age")
            train_d_ohe = enc.fit_transform(train_d_res, train_l_res)
            train_d_res, train_l_res = us.fit_resample(train_d_ohe, train_l_res)
            cat_part = train_d_res[:, :-1]
            cont_part = train_d_res[:, -1:]
            train_d_cont = pd.DataFrame((enc.named_transformers_["AgeScaler"].inverse_transform(cont_part)), columns=['age'])
            train_d_cont = train_d_cont.astype(int)
            train_d_res = pd.DataFrame((enc.named_transformers_["OneHot"].inverse_transform(cat_part)), columns=cat_names)
            train_d_res.insert(age_index, "age", train_d_cont)
        else:
            train_d_ohe = enc.fit_transform(train_d_res, train_l_res)
            train_d_res, train_l_res = us.fit_resample(train_d_ohe, train_l_res)
            train_d_res = pd.DataFrame((enc.named_transformers_["OneHot"].inverse_transform(train_d_res)), columns=columns)
    return train_d_res, train_l_res


def count_labels(labels):
    labels_copy = labels.copy()
    labels_df = pd.DataFrame(labels_copy, columns=["diag_multi"])
    resampled_counts = labels_df["diag_multi"].value_counts()
    print("Resampled label counts:")
    print(resampled_counts)
    return resampled_counts
