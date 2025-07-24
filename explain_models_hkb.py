import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import explain_utils
import feature_loader
import metrics
import utils


def make_folders(path):
    os.makedirs(f"./output/{path}/shap/plots", exist_ok=True)
    os.makedirs(f"./output/{path}/lime/plots", exist_ok=True)


results_path_hkb = "hkb_explain"
results_path_hkb_res = "hkb_explain_res"

make_folders(results_path_hkb)
make_folders(results_path_hkb_res)

seed = np.random.randint(0, 2 ** 31 - 1)

data = pd.read_csv("../data/data.csv", header=0)
X = data[feature_loader.no_lab_features]
y = data[['diag_multi']]
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=1040605528)

baseline = metrics.get_baseline(X)
explain_data = test_data
explain_data_res, explain_labels_res = utils.subsample_data(test_data, test_labels, seed)
background_data = train_data
background_data_res, background_labels_res = utils.subsample_data(train_data, train_labels, seed)
feature_names = feature_loader.no_lab_features

hkb_results_shap, hkb_results_lime, _, _ = explain_utils.fi_explain(
    "hkb", explain_data, background_data, feature_names, baseline, seed, True, "hkb_0.kb",
    results_path_hkb, "hkb_0_discretized.sa",
    "temp_samples.txt", "temp_predictions.txt"
)

with open(f"./output/{results_path_hkb}/explain_results_hkb_shap.json", "w") as f:
    json.dump(hkb_results_shap, f, indent=4)
with open(f"./output/{results_path_hkb}/explain_results_hkb_lime.json", "w") as f:
    json.dump(hkb_results_lime, f, indent=4)


hkb_results_shap2, hkb_results_lime2, _, _ = explain_utils.fi_explain(
    "hkb", explain_data, background_data_res, feature_names, baseline, seed, True, "hkb_0.kb",
    results_path_hkb_res, "hkb_0_discretized.sa",
    "temp_samples.txt", "temp_predictions.txt"
)

with open(f"./output/{results_path_hkb_res}/explain_results_hkb_shap_res.json", "w") as f:
    json.dump(hkb_results_shap2, f, indent=4)
with open(f"./output/{results_path_hkb_res}/explain_results_hkb_lime_res.json", "w") as f:
    json.dump(hkb_results_lime2, f, indent=4)
