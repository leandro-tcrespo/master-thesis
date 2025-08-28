import json
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.under_sampling import ClusterCentroids

import feature_loader
import explain_utils
import metrics
import preprocessing
import utils


def make_folders(path):
    if "hkb" not in path:
        os.makedirs(f"./output/{path}/shap/plots", exist_ok=True)
        os.makedirs(f"./output/{path}/lime/plots", exist_ok=True)
    if "bg" not in path and "both" not in path and "mlp" not in path:
        os.makedirs(f"./output/{path}/model_exp/plots", exist_ok=True)
    os.makedirs("./explain_data_backups", exist_ok=True)


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

with open("dt.pkl", "rb") as file:
    dt = pickle.load(file)

with open("mlp.pkl", "rb") as file:
    mlp = pickle.load(file)

train_data, test_data, train_labels, test_labels, enc, _, _, _, _ = preprocessing.preprocess_data("./Synthetic_data.csv", feature_loader.all_features, seed)

# This undersampling was specific to the given explain and background datasets.
# # "Hardcode" resampling, so "Kein" samples are subsampled to same sample size as "RA".
# explain_resampler = ClusterCentroids(sampling_strategy={"Kein": 23, "SpA":8, "PsA":23, "RA":22}, random_state=seed, voting="hard")
# background_resampler = ClusterCentroids(sampling_strategy={"Kein": 74, "SpA":25, "PsA":54, "RA":74}, random_state=seed, voting="hard")

# Undersampling for demonstration purposes.
explain_resampler = ClusterCentroids(random_state=seed, voting="hard")
background_resampler = ClusterCentroids(random_state=seed, voting="hard")

# Set up the baseline for faithfulness correlation and the explain data.
baseline = metrics.get_baseline(pd.concat([train_data, test_data]))
explain_data = test_data

# Note that this fits the encoder on the test data. This is necessary here since the main goal is to correctly represent
# the feature diversity in the test data, this would not necessarily be the case if the encoder fit on train data would
# be used since there may be features that appear in the test data but not in the train data, those features would be ignored.
explain_data_res, explain_labels_res = utils.resample_data(enc, test_data, test_labels, us=explain_resampler)
background_data = train_data
background_data_res, background_labels_res = utils.resample_data(enc, train_data, train_labels, us=background_resampler)
feature_names = feature_loader.all_features

# Count labels for logging.
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
