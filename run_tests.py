import os

import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
import metrics
from lime import lime_tabular
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import dt
import hkb
import lime_utils
import mlp
import preprocessing

os.makedirs('./output', exist_ok=True)


def predict_proba_wrapper_mlp(original_data):
    original_data_copy = pd.DataFrame(original_data, columns=train_data.columns)
    return grid_mlp_smote.predict_proba(original_data_copy)


def predict_proba_wrapper_hkb(original_data):
    original_data_df = pd.DataFrame(original_data, columns=train_data.columns)
    non_age_cols = original_data_df.columns.difference(['age'])
    # this is necessary because the data from Lime is of type float, this is a problem for
    # properly formatting categorical features (e.g. a_2.0 instead of a_2)
    original_data_df[non_age_cols] = original_data_df[non_age_cols].astype(int)
    return hkb.predict_proba(original_data_df, './output/lime_predictions_hkb.txt')


train_data, test_data, train_labels, test_labels, enc, smote_os, random_os = preprocessing.preprocess_data("./Synthetic_data.csv")
train_data_lime = train_data.copy()
train_data_lime = train_data_lime.to_numpy()

################################################
# MLP testing
################################################

base_mlp = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
smote_pipeline = make_pipeline(smote_os, enc, base_mlp)
randomos_pipeline = make_pipeline(random_os, enc, base_mlp)

print("Starting SMOTE MLP Training")
grid_mlp_smote = mlp.fit(smote_pipeline, train_data, train_labels)
predictions_mlp = grid_mlp_smote.predict(test_data)
print("SMOTE results:")
metrics.score(test_labels, predictions_mlp)

print("Starting RandomOS MLP Training")
grid_mlp_randomos = mlp.fit(randomos_pipeline, train_data, train_labels)
predictions_mlp = grid_mlp_randomos.predict(test_data)
print("RandomOS results:")
metrics.score(test_labels, predictions_mlp)

# # todo add categorical names to make lime explanations more understandable,
# #  probably have to convert features 1,2,3,9 to 0,1,2,3 because
# #  categorical names dict is accessed by looking at index indicated by feature value i in column x: names[x][i]
# explainer_mlp = lime_tabular.LimeTabularExplainer(
#     train_data_lime,
#     categorical_features=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     feature_names=['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'],
#     class_names=grid_mlp_smote.classes_.tolist(),
#     verbose=True,
#     random_state=42
# )
# exp_mlp = explainer_mlp.explain_instance(test_data.values[0], predict_proba_wrapper_mlp, top_labels=1, num_features=19)
# fi_values_dict = exp_mlp.as_list(exp_mlp.available_labels()[0])
# fi_values = np.array([fi_value for _,fi_value in fi_values_dict])
# print(metrics.compute_complexity(fi_values))
# baseline = [1]*19
# baseline = np.asarray(baseline)
# print(metrics.faithfulness_corr(model=grid_mlp_smote, input_sample=test_data.values[0], feature_names=['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'], attributions=fi_values, baseline=baseline, label=0))
# fig = lime_utils.plot_lime(exp_mlp, './output/lime_plot_mlp.png')
# fig.show()

################################################
# HKB testing
################################################

hkb.fit(train_data, train_labels)
predictions = hkb.predict(test_data, './output/predictions.txt')
print("HKB results:")
metrics.score(test_labels, predictions)

# explainer_hkb = lime_tabular.LimeTabularExplainer(
#     train_data_lime,
#     categorical_features=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     feature_names=['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'],
#     class_names=["Kein", "RA", "SpA", "PsA"],
#     verbose=True,
#     random_state=42
# )
# exp_hkb = explainer_hkb.explain_instance(test_data.values[0], predict_proba_wrapper_hkb, top_labels=1, num_samples=100)
# fig = lime_utils.plot_lime(exp_hkb, './output/lime_plot_hkb.png')
# fig.show()

################################################
# DT testing
################################################

base_dt = DecisionTreeClassifier(random_state=42)
dt_pipeline = make_pipeline(smote_os, enc, base_dt)

print("Starting DT Training")
grid_dt = dt.fit(dt_pipeline, train_data, train_labels)
predictions_dt = grid_dt.predict(test_data)
print("DT results:")
metrics.score(test_labels, predictions_dt)
dt.plot_dt(grid_dt)
