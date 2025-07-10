import json
import os
import pickle

import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
from sklearn.model_selection import train_test_split

import feature_loader
import hkb
import explain_utils
import metrics
import utils


def make_folders(path):
    os.makedirs(f"./output/{path}/shap/plots", exist_ok=True)
    os.makedirs(f"./output/{path}/lime/plots", exist_ok=True)


results_path_mlp = "mlp_explain"
results_path_dt = "dt_explain"
# results_path_hkb = "hkb_explain"
results_path_mlp_res = "mlp_explain_res"
results_path_dt_res = "dt_explain_res"
# results_path_hkb_res = "hkb_explain_res"

make_folders(results_path_mlp)
make_folders(results_path_dt)
# make_folders(results_path_hkb)
make_folders(results_path_mlp_res)
make_folders(results_path_dt_res)
# make_folders(results_path_hkb_res)

seed = np.random.randint(0, 2 ** 31 - 1)

with open('dt_0.pkl', 'rb') as file:
    dt = pickle.load(file)

with open('mlp_0.pkl', 'rb') as file:
    mlp = pickle.load(file)

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

background_label_count = utils.count_labels(train_labels).to_json()
background_label_res_count = utils.count_labels(background_labels_res).to_json()
explain_label_count = utils.count_labels(test_labels).to_json()
explain_label_res_count = utils.count_labels(explain_labels_res).to_json()

label_counts = {"Background data:": background_label_count,
                "Background data resampled:": background_label_res_count,
                "Explain data:": explain_label_count,
                "Explain data resampled:": explain_label_res_count}

with open(f"./output/label_counts.json", "w") as f:
    json.dump(label_counts, f, indent=4)

mlp_results_shap, mlp_results_lime, _, _ = explain_utils.explain_model(
    mlp, explain_data, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp
)

with open(f"./output/{results_path_mlp}/explain_results_mlp_shap.json", "w") as f:
    json.dump(mlp_results_shap, f, indent=4)
with open(f"./output/{results_path_mlp}/explain_results_mlp_lime.json", "w") as f:
    json.dump(mlp_results_lime, f, indent=4)


mlp_results_shap2, mlp_results_lime2, _, _ = explain_utils.explain_model(
    mlp, explain_data, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp_res
)

with open(f"./output/{results_path_mlp_res}/explain_results_mlp_shap_res.json", "w") as f:
    json.dump(mlp_results_shap2, f, indent=4)
with open(f"./output/{results_path_mlp_res}/explain_results_mlp_lime_res.json", "w") as f:
    json.dump(mlp_results_lime2, f, indent=4)


dt_results_shap, dt_results_lime, _, _ = explain_utils.explain_model(
    dt, explain_data, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt
)

with open(f"./output/{results_path_dt}/explain_results_dt_shap.json", "w") as f:
    json.dump(dt_results_shap, f, indent=4)
with open(f"./output/{results_path_dt}/explain_results_dt_lime.json", "w") as f:
    json.dump(dt_results_lime, f, indent=4)

dt_results_shap2, dt_results_lime2, _, _ = explain_utils.explain_model(
    dt, explain_data, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt_res
)

with open(f"./output/{results_path_dt_res}/explain_results_dt_shap_res.json", "w") as f:
    json.dump(dt_results_shap2, f, indent=4)
with open(f"./output/{results_path_dt_res}/explain_results_dt_lime_res.json", "w") as f:
    json.dump(dt_results_lime2, f, indent=4)
