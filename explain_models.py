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
    if "hkb" not in path:
        os.makedirs(f"./output/{path}/shap/plots", exist_ok=True)
        os.makedirs(f"./output/{path}/lime/plots", exist_ok=True)
    if "bg" not in path and "both" not in path and "mlp" not in path:
        os.makedirs(f"./output/{path}/model_exp/plots", exist_ok=True)


results_path_mlp = "mlp_exp"
results_path_dt = "dt_exp"
results_path_hkb = "hkb_exp"

results_path_mlp_bg_res = "mlp_exp_bg_res"
results_path_dt_bg_res = "dt_exp_bg_res"

results_path_mlp_data_res = "mlp_exp_data_res"
results_path_dt_data_res = "dt_exp_data_res"
results_path_hkb_data_res = "hkb_exp_data_res"

results_path_mlp_both_res = "mlp_exp_both_res"
results_path_dt_both_res = "dt_exp_both_res"

make_folders(results_path_mlp)
make_folders(results_path_dt)
make_folders(results_path_hkb)

make_folders(results_path_mlp_bg_res)
make_folders(results_path_dt_bg_res)

make_folders(results_path_mlp_data_res)
make_folders(results_path_dt_data_res)
make_folders(results_path_hkb_data_res)

make_folders(results_path_mlp_both_res)
make_folders(results_path_dt_both_res)

seed = np.random.randint(0, 2 ** 31 - 1)

with open('dt.pkl', 'rb') as file:
    dt = pickle.load(file)

with open('mlp.pkl', 'rb') as file:
    mlp = pickle.load(file)

data = pd.read_csv("../data/data.csv", header=0)
X = data[feature_loader.no_lab_features]
y = data[['diag_multi']]
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=598398916)

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

########################################################################################################################
# MLP FI Testing
########################################################################################################################

mlp_results_shap, mlp_results_lime, _, _ = explain_utils.fi_explain(
    mlp, explain_data, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp
)

with open(f"./output/{results_path_mlp}/explain_results_mlp_shap.json", "w") as f:
    json.dump(mlp_results_shap, f, indent=4)
with open(f"./output/{results_path_mlp}/explain_results_mlp_lime.json", "w") as f:
    json.dump(mlp_results_lime, f, indent=4)


mlp_results_shap2, mlp_results_lime2, _, _ = explain_utils.fi_explain(
    mlp, explain_data, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp_bg_res
)

with open(f"./output/{results_path_mlp_bg_res}/explain_results_mlp_shap_bg_res.json", "w") as f:
    json.dump(mlp_results_shap2, f, indent=4)
with open(f"./output/{results_path_mlp_bg_res}/explain_results_mlp_lime_bg_res.json", "w") as f:
    json.dump(mlp_results_lime2, f, indent=4)

mlp_results_shap3, mlp_results_lime3, _, _ = explain_utils.fi_explain(
    mlp, explain_data_res, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp_data_res
)

with open(f"./output/{results_path_mlp_data_res}/explain_results_mlp_shap_data_res.json", "w") as f:
    json.dump(mlp_results_shap3, f, indent=4)
with open(f"./output/{results_path_mlp_data_res}/explain_results_mlp_lime_data_res.json", "w") as f:
    json.dump(mlp_results_lime3, f, indent=4)

mlp_results_shap4, mlp_results_lime4, _, _ = explain_utils.fi_explain(
    mlp, explain_data_res, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_mlp_both_res
)

with open(f"./output/{results_path_mlp_both_res}/explain_results_mlp_shap_both_res.json", "w") as f:
    json.dump(mlp_results_shap4, f, indent=4)
with open(f"./output/{results_path_mlp_both_res}/explain_results_mlp_lime_both_res.json", "w") as f:
    json.dump(mlp_results_lime4, f, indent=4)

########################################################################################################################
# DT FI Testing
########################################################################################################################

dt_results_shap, dt_results_lime, _, _ = explain_utils.fi_explain(
    dt, explain_data, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt
)

with open(f"./output/{results_path_dt}/explain_results_dt_shap.json", "w") as f:
    json.dump(dt_results_shap, f, indent=4)
with open(f"./output/{results_path_dt}/explain_results_dt_lime.json", "w") as f:
    json.dump(dt_results_lime, f, indent=4)

dt_results_shap2, dt_results_lime2, _, _ = explain_utils.fi_explain(
    dt, explain_data, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt_bg_res
)

with open(f"./output/{results_path_dt_bg_res}/explain_results_dt_shap_bg_res.json", "w") as f:
    json.dump(dt_results_shap2, f, indent=4)
with open(f"./output/{results_path_dt_bg_res}/explain_results_dt_lime_bg_res.json", "w") as f:
    json.dump(dt_results_lime2, f, indent=4)

dt_results_shap3, dt_results_lime3, _, _ = explain_utils.fi_explain(
    dt, explain_data_res, background_data, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt_data_res
)

with open(f"./output/{results_path_dt_data_res}/explain_results_dt_shap_data_res.json", "w") as f:
    json.dump(dt_results_shap3, f, indent=4)
with open(f"./output/{results_path_dt_data_res}/explain_results_dt_lime_data_res.json", "w") as f:
    json.dump(dt_results_lime3, f, indent=4)

dt_results_shap4, dt_results_lime4, _, _ = explain_utils.fi_explain(
    dt, explain_data_res, background_data_res, feature_names, baseline, seed, plot_explanations=True, name=results_path_dt_both_res
)

with open(f"./output/{results_path_dt_both_res}/explain_results_dt_shap_both_res.json", "w") as f:
    json.dump(dt_results_shap4, f, indent=4)
with open(f"./output/{results_path_dt_both_res}/explain_results_dt_lime_both_res.json", "w") as f:
    json.dump(dt_results_lime4, f, indent=4)

########################################################################################################################
# DT Model explanation testing
########################################################################################################################

dt_results_model_exp = explain_utils.model_explain(dt, explain_data, results_path_dt)

with open(f"./output/{results_path_dt}/explain_results_dt_model_exp.json", "w") as f:
    json.dump(dt_results_model_exp, f, indent=4)

dt_results_model_exp2 = explain_utils.model_explain(dt, explain_data_res, results_path_dt_data_res)

with open(f"./output/{results_path_dt_data_res}/explain_results_dt_model_exp.json", "w") as f:
    json.dump(dt_results_model_exp2, f, indent=4)

########################################################################################################################
# HKB Model explanation testing
########################################################################################################################

hkb_results_model_exp = explain_utils.model_explain("hkb.kb", explain_data, results_path_hkb)

with open(f"./output/{results_path_hkb}/explain_results_hkb_model_exp.json", "w") as f:
    json.dump(hkb_results_model_exp, f, indent=4)

hkb_results_model_exp2 = explain_utils.model_explain("hkb.kb", explain_data_res, results_path_hkb_data_res)

with open(f"./output/{results_path_hkb_data_res}/explain_results_hkb_model_exp.json", "w") as f:
    json.dump(hkb_results_model_exp2, f, indent=4)

########################################################################################################################
# One-Time Prediction for HKB on whole Train Data to check how many "multi" predictions InteKRator makes
########################################################################################################################
hkb.predict(X, "", "hkb.kb", "temp_formatted_samples.txt", "./output/all_hkb_preds.txt")
