import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from sklearn import clone
from sklearn.model_selection import train_test_split

import hkb


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


def subsample_data(data, labels, seed):
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=seed) # todo: include in thesis that random undersampling is not optimal since samples with "unclear" decision boundaries could be chosen, clustering would be better but would have to be treated with care because of the categorical features
    data_res, labels_res = rus.fit_resample(data, labels)
    return data_res, labels_res


def get_predicted_labels(model, data, name=None, kb=None, formatted_samples_path=None, pred_out=None):
    if model == "hkb":
        class_names = hkb.CLASS_ORDER
        preds = hkb.predict(data, name, kb, formatted_samples_path, pred_out)
        predicted_inds = [hkb.CLASS_ORDER.index(pred) for pred in preds]
    else:
        class_names = model.steps[-1][1].classes_
        preds = model.predict_proba(data)
        predicted_inds = np.argmax(preds, axis=1)
    predicted_strings = [class_names[i] for i in predicted_inds]
    return predicted_inds, predicted_strings
