import os

import utils

# Suppresses warnings about onednn custom operations being on and available CPU instructions for potential better perf.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.base import clone

import kerasmlp
import metrics
from sklearn.tree import DecisionTreeClassifier

import dt
import hkb
import preprocessing

os.makedirs("./output/tuning_results/MLP", exist_ok=True)
os.makedirs("./output/tuning_results/HKB", exist_ok=True)
os.makedirs("./output/tuning_results/DT", exist_ok=True)


def set_output(name):
    f = open(f"./output/tuning_results/{name}.txt", "w")
    return f


# For each tested parameter take all mean test scores of param configs it was active in and calculate the mean of those.
# This gives a rough estimate of how good a parameter performed during the CV
# The code skeleton for this method was created by an LLM, see thesis for details on the prompt and version.
def mean_scores_cv(model, results, output):
    summary = []
    for param, values in model.param_grid.items():
        for value in values:
            # Get the list with the mean test scores for each parameter configuration and filter it, so we store only
            # the values for the corresponding parameter, the boolean indexing mask is created by the param_... list
            # that contains the active parameter values for each tested configuration
            # example:
            # mean_test_score = [0.5, 0.4, 0.3, 0.4] indicates four tested parameter configs
            # and param_num_layers = [1, 1, 2, 2] indicates the active values for num_layers in each tested config
            # for value=1 we get the mask [True, True, False, False], so the stored scores are [0.5, 0.4]
            # for more details on cv_results_ refer to:
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            scores = results["mean_test_score"][results[f"param_{param}"] == value]
            param_name = param.replace("kerasclassifier__", "")
            summary.append((f"{param_name}_{value}", np.mean(scores)))

    summary_df = pd.DataFrame(summary, columns=["Parameter", "Mean Score"])
    summary_df.to_csv(output, index=False)


seed = 42
all_features = ["sex", "age", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q"]
train_data, test_data, train_labels, test_labels, enc, ros, tomek, smote, enn = (preprocessing.preprocess_data("./Synthetic_data.csv", all_features, seed))

train_data_smote, train_labels_smote = utils.resample_data(enc, train_data, train_labels, smote)
train_data_smote_enn, train_labels_smote_enn = utils.resample_data(enc, train_data, train_labels, smote, enn)
train_data_smote_tomek, train_labels_smote_tomek = utils.resample_data(enc, train_data, train_labels, smote, tomek)
train_data_ros_tomek, train_labels_ros_tomek = utils.resample_data(enc, train_data, train_labels, ros, tomek)

print("\nNo Resampling:")
utils.count_labels(train_labels)
print("\nSMOTE:")
utils.count_labels(train_labels_smote)
print("\nSMOTEENN:")
utils.count_labels(train_labels_smote_enn)
print("\nSMOTETOMEK:")
utils.count_labels(train_labels_smote_tomek)
print("\nROS+TOMEK:")
utils.count_labels(train_labels_ros_tomek)

########################################################################################################################
# MLP Hyperparameter-Tuning
########################################################################################################################

base_mlp = kerasmlp.get_keras_model()
smote_pipeline_baseline = make_pipeline(clone(enc), clone(base_mlp))
smote_pipeline = make_pipeline(clone(smote), clone(enc), clone(base_mlp))
smote_enn_pipeline = make_pipeline(clone(smote), clone(enc), clone(enn), clone(base_mlp))
smote_tomek_pipeline = make_pipeline(clone(smote), clone(enc), clone(tomek), clone(base_mlp))
ros_tomek_pipeline = make_pipeline(clone(ros), clone(enc), clone(tomek), clone(base_mlp))

print("Starting Baseline MLP Training (only ohe and age scaling)...")
grid_mlp_baseline = kerasmlp.fit(smote_pipeline_baseline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_baseline.best_params_}")
baseline_mlp = grid_mlp_baseline.predict(test_data)
metrics.score(test_labels, baseline_mlp, "Baseline_MLP", set_output("MLP/baseline_mlp_results"))

print("Starting SMOTE MLP Training...")
grid_mlp_smote = kerasmlp.fit(smote_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_smote.best_params_}")
mean_scores_cv(grid_mlp_smote, grid_mlp_smote.cv_results_, "./output/tuning_results/MLP/mlp_cv_summary_smote.csv")
predictions_mlp = grid_mlp_smote.predict(test_data)
metrics.score(test_labels, predictions_mlp, "SMOTE_MLP", set_output("MLP/smote_mlp_results"))

print("Starting SMOTEENN MLP Training...")
grid_mlp_smote_enn = kerasmlp.fit(smote_enn_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_smote_enn.best_params_}")
mean_scores_cv(grid_mlp_smote_enn, grid_mlp_smote_enn.cv_results_, "./output/tuning_results/MLP/mlp_cv_summary_smoteenn.csv")
smote_enn_mlp = grid_mlp_smote_enn.predict(test_data)
metrics.score(test_labels, smote_enn_mlp, "SMOTEENN_MLP", set_output("MLP/smoteenn_mlp_results"))

print("Starting SMOTETOMEK MLP Training...")
grid_mlp_smote_tomek = kerasmlp.fit(smote_tomek_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_smote_tomek.best_params_}")
mean_scores_cv(grid_mlp_smote_tomek, grid_mlp_smote_tomek.cv_results_, "./output/tuning_results/MLP/mlp_cv_summary_smotetomek.csv")
smote_tomek_mlp = grid_mlp_smote_tomek.predict(test_data)
metrics.score(test_labels, smote_tomek_mlp, "SMOTETOMEK_MLP", set_output("MLP/smotetomek_mlp_results"))

print("Starting ROSTOMEK MLP Training...")
grid_mlp_ros_tomek = kerasmlp.fit(ros_tomek_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_ros_tomek.best_params_}")
mean_scores_cv(grid_mlp_ros_tomek, grid_mlp_ros_tomek.cv_results_, "./output/tuning_results/MLP/mlp_cv_summary_rostomek.csv")
ros_tomek_mlp = grid_mlp_ros_tomek.predict(test_data)
metrics.score(test_labels, ros_tomek_mlp, "ROSTOMEK_MLP", set_output("MLP/rostomek_mlp_results"))

########################################################################################################################
# HKB Hyperparameter-Tuning of 2,3 or 4 clusters for HKBs, done "manually".
########################################################################################################################

############################
# 2 Clusters
############################

hkb.fit(train_data, train_labels, "", "2}2", "hkb_2.kb")
predictions = hkb.predict(test_data, "", "hkb_2.kb", "temp_samples.txt", "./output/tuning_results/HKB/predictions_2.txt")
metrics.score(test_labels, predictions, "Baseline_HKB_2", set_output("MLP/baseline_hkb_2_results"))

hkb.fit(train_data_smote, train_labels_smote, "", "2}2", "smote_2.kb")
predictions = hkb.predict(test_data, "", "smote_2.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_2.txt")
metrics.score(test_labels, predictions, "HKB_SMOTE_2", set_output("HKB/hkb_smote_2_results"))

hkb.fit(train_data_smote_enn, train_labels_smote_enn, "", "2}2", "smote_enn_2.kb")
predictions = hkb.predict(test_data, "", "smote_enn_2.kb", "temp_samples.txt", "./output/tuning_results/HKB/predictions_smote_enn_2.txt")
metrics.score(test_labels, predictions, "HKB_SMOTEENN_2", set_output("HKB/hkb_smoteenn_2_results"))

hkb.fit(train_data_smote_tomek, train_labels_smote_tomek, "", "2}2", "smote_tomek_2.kb")
predictions = hkb.predict(test_data, "", "smote_tomek_2.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_tomek_2.txt")
metrics.score(test_labels, predictions, "HKB_SMOTETOMEK_2", set_output("HKB/hkb_smotetomek_2_results"))

hkb.fit(train_data_ros_tomek, train_labels_ros_tomek, "", "2}2", "ros_tomek_2.kb")
predictions = hkb.predict(test_data, "", "ros_tomek_2.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_ros_tomek_2.txt")
metrics.score(test_labels, predictions, "HKB_ROSTOMEK_2", set_output("HKB/hkb_rostomek_2_results"))

############################
# 3 Clusters
############################

hkb.fit(train_data, train_labels, "", "2}3", "hkb_3.kb")
predictions = hkb.predict(test_data, "", "hkb_3.kb", "temp_samples.txt", "./output/tuning_results/HKB/predictions_3.txt")
metrics.score(test_labels, predictions, "Baseline_HKB_3", set_output("MLP/baseline_hkb_3_results"))

hkb.fit(train_data_smote, train_labels_smote, "", "2}3", "smote_3.kb")
predictions = hkb.predict(test_data, "", "smote_3.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_3.txt")
metrics.score(test_labels, predictions, "HKB_SMOTE_3", set_output("HKB/hkb_smote_3_results"))

hkb.fit(train_data_smote_enn, train_labels_smote_enn, "", "2}3", "smote_enn_3.kb")
smoteenn_predictions = hkb.predict(test_data, "", "smote_enn_3.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smoteenn_3.txt")
metrics.score(test_labels, smoteenn_predictions, "HKB_SMOTEENN_3", set_output("HKB/hkb_smoteenn_3_results"))

hkb.fit(train_data_smote_tomek, train_labels_smote_tomek, "", "2}3", "smote_tomek_3.kb")
smotetomek_predictions = hkb.predict(test_data, "", "smote_tomek_3.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smotetomek_3.txt")
metrics.score(test_labels, smotetomek_predictions, "HKB_SMOTETOMEK_3", set_output("HKB/hkb_smotetomek_3_results"))

hkb.fit(train_data_ros_tomek, train_labels_ros_tomek, "", "2}3", "ros_tomek_3.kb")
rostomek_predictions = hkb.predict(test_data, "", "ros_tomek_3.kb", "temp_samples.txt", "./output/tuning_results/HKB/predictions_rostomek_3.txt")
metrics.score(test_labels, rostomek_predictions, "HKB_ROSTOMEK_3", set_output("HKB/hkb_rostomek_3_results"))

############################
# 4 Clusters
############################

hkb.fit(train_data, train_labels, "", "2}4", "hkb_4.kb")
predictions = hkb.predict(test_data, "", "hkb_4.kb", "temp_samples.txt", "./output/tuning_results/HKB/predictions_4.txt")
metrics.score(test_labels, predictions, "Baseline_HKB_4", set_output("MLP/baseline_hkb_4_results"))

hkb.fit(train_data_smote, train_labels_smote, "", "2}4", "smote_4.kb")
predictions = hkb.predict(test_data, "", "smote_4.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_4.txt")
metrics.score(test_labels, predictions, "HKB_SMOTE_4", set_output("HKB/hkb_smote_4_results"))

hkb.fit(train_data_smote_enn, train_labels_smote_enn, "", "2}4", "smote_enn_4.kb")
predictions = hkb.predict(test_data, "", "smote_enn_4.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_enn_4.txt")
metrics.score(test_labels, predictions, "HKB_SMOTEENN_4", set_output("HKB/hkb_smoteenn_4_results"))

hkb.fit(train_data_smote_tomek, train_labels_smote_tomek, "", "2}4", "smote_tomek_4.kb")
predictions = hkb.predict(test_data, "", "smote_tomek_4.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_smote_tomek_4.txt")
metrics.score(test_labels, predictions, "HKB_SMOTETOMEK_4", set_output("HKB/hkb_smotetomek_4_results"))

hkb.fit(train_data_ros_tomek, train_labels_ros_tomek, "", "2}4", "ros_tomek_4.kb")
predictions = hkb.predict(test_data, "", "ros_tomek_4.kb", "temp_samples.txt",  "./output/tuning_results/HKB/predictions_ros_tomek_4.txt")
metrics.score(test_labels, predictions, "HKB_ROSTOMEK_4", set_output("HKB/hkb_rostomek_4_results"))

########################################################################################################################
# DT Hyperparameter-Tuning with CCP-Alpha Pruning.
########################################################################################################################

base_dt = DecisionTreeClassifier()

base_pipeline_dt = make_pipeline(clone(enc), clone(base_dt))
smote_pipeline_dt = make_pipeline(clone(smote), clone(enc), clone(base_dt))
smote_enn_pipeline_dt = make_pipeline(clone(smote), clone(enc), clone(enn), clone(base_dt))
smote_tomek_pipeline_dt = make_pipeline(clone(smote), clone(enc), clone(tomek), clone(base_dt))
ros_tomek_pipeline_dt = make_pipeline(clone(ros), clone(enc), clone(tomek), clone(base_dt))

print("Starting DT Baseline Training...")
grid_base_dt = dt.fit(base_pipeline_dt, train_data, train_labels, dt.param_grid)
print(f"Training finished, best params: {grid_base_dt.best_params_}")
predictions_base_dt = grid_base_dt.predict(test_data)
metrics.score(test_labels, predictions_base_dt, "DT_Baseline", set_output("DT/dt_baseline_results"))
dt.plot_dt(grid_base_dt, "./output/tuning_results/DT/grid_base_dt.png")

print("Searching for ccp alpha values...")
grid_base_ccp = dt.prune(grid_base_dt.best_estimator_, train_data, train_labels)
predictions_base_ccp = grid_base_ccp.predict(test_data)
metrics.score(test_labels, predictions_base_ccp, "Baseline_Pruned", set_output("DT/base_ccp_results"))
dt.plot_dt(grid_base_ccp, "./output/tuning_results/DT/grid_base_ccp.png")


print("Starting DT Training with SMOTE...")
grid_smote_dt = dt.fit(smote_pipeline_dt, train_data, train_labels, dt.param_grid)
print(f"Training finished, best params: {grid_smote_dt.best_params_}")
mean_scores_cv(grid_smote_dt, grid_smote_dt.cv_results_, "./output/tuning_results/DT/dt_cv_summary_smote.csv")
predictions_smote_dt = grid_smote_dt.predict(test_data)
metrics.score(test_labels, predictions_smote_dt, "DT_SMOTE", set_output("dt_smote_results"))
dt.plot_dt(grid_smote_dt, "./output/tuning_results/DT/grid_smote_dt.png")

print("Searching for ccp alpha values...")
grid_smote_ccp = dt.prune(grid_smote_dt.best_estimator_, train_data, train_labels)
predictions_smote_ccp = grid_smote_ccp.predict(test_data)
metrics.score(test_labels, predictions_smote_ccp, "SMOTE_Pruned", set_output("DT/smote_ccp_results"))
dt.plot_dt(grid_smote_ccp, "./output/tuning_results/DT/grid_smote_ccp.png")


print("Starting SMOTEENN DT Training...")
grid_smote_enn_dt = dt.fit(smote_enn_pipeline_dt, train_data, train_labels, dt.param_grid)
print(f"Training finished, best params: {grid_smote_enn_dt.best_params_}")
mean_scores_cv(grid_smote_enn_dt, grid_smote_enn_dt.cv_results_, "./output/tuning_results/DT/dt_cv_summary_smoteenn.csv")
predictions_smote_enn_dt = grid_smote_enn_dt.predict(test_data)
metrics.score(test_labels, predictions_smote_enn_dt, "DT_SMOTEENN", set_output("dt_smoteenn_results"))
dt.plot_dt(grid_smote_enn_dt, "./output/tuning_results/DT/grid_smote_enn_dt.png")

print("Searching for ccp alpha values...")
grid_smote_enn_ccp = dt.prune(grid_smote_enn_dt.best_estimator_, train_data, train_labels)
predictions_smote_enn_ccp = grid_smote_enn_ccp.predict(test_data)
metrics.score(test_labels, predictions_smote_enn_ccp, "SMOTEENN_Pruned", set_output("DT/smoteenn_ccp_results"))
dt.plot_dt(grid_smote_enn_ccp, "./output/tuning_results/DT/grid_smote_enn_ccp.png")


print("Starting SMOTETOMEK DT Training...")
grid_smote_tomek_dt = dt.fit(smote_tomek_pipeline_dt, train_data, train_labels, dt.param_grid)
print(f"Training finished, best params: {grid_smote_tomek_dt.best_params_}")
mean_scores_cv(grid_smote_tomek_dt, grid_smote_tomek_dt.cv_results_, "./output/tuning_results/DT/dt_cv_summary_smotetomek.csv")
predictions_smote_tomek_dt = grid_smote_tomek_dt.predict(test_data)
metrics.score(test_labels, predictions_smote_tomek_dt, "DT_SMOTETOMEK", set_output("DT/dt_smotetomek_results"))
dt.plot_dt(grid_smote_tomek_dt, "./output/tuning_results/DT/grid_smote_tomek_dt.png")

print("Searching for ccp alpha values...")
grid_smote_tomek_ccp = dt.prune(grid_smote_tomek_dt.best_estimator_, train_data, train_labels)
predictions_smote_tomek_ccp = grid_smote_tomek_ccp.predict(test_data)
metrics.score(test_labels, predictions_smote_tomek_ccp, "SMOTETOMEK_Pruned", set_output("DT/smotetomek_ccp_results"))
dt.plot_dt(grid_smote_tomek_ccp, "./output/tuning_results/DT/grid_smote_tomek_ccp.png")


print("Starting ROSTOMEK DT Training...")
grid_ros_tomek_dt = dt.fit(ros_tomek_pipeline_dt, train_data, train_labels, dt.param_grid)
print(f"Training finished, best params: {grid_ros_tomek_dt.best_params_}")
mean_scores_cv(grid_ros_tomek_dt, grid_ros_tomek_dt.cv_results_, "./output/tuning_results/DT/dt_cv_summary_rostomek.csv")
predictions_ros_tomek_dt = grid_ros_tomek_dt.predict(test_data)
metrics.score(test_labels, predictions_ros_tomek_dt, "DT_ROSTOMEK", set_output("DT/dt_rostomek_results"))
dt.plot_dt(grid_ros_tomek_dt, "./output/tuning_results/DT/grid_ros_tomek_dt.png")

print("Searching for ccp alpha values...")
grid_ros_tomek_ccp = dt.prune(grid_ros_tomek_dt.best_estimator_, train_data, train_labels)
predictions_ros_tomek_ccp = grid_ros_tomek_ccp.predict(test_data)
metrics.score(test_labels, predictions_ros_tomek_ccp, "ROSTOMEK_Pruned", set_output("DT/rostomek_ccp_results"))
dt.plot_dt(grid_ros_tomek_ccp, "./output/tuning_results/DT/grid_ros_tomek_ccp.png")
