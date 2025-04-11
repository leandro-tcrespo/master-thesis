import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import kerasmlp
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


# for each tested parameter take all mean test scores of param configs it was active in and calculate the mean of those
# this gives a rough estimate of how good a parameter performed during the CV
def mean_scores_cv(model, results, output):
    summary = []
    for param, values in model.param_grid.items():
        for value in values:
            # get the list with the mean test scores for each parameter configuration and filter it, so we store only
            # the values for the corresponding parameter, the boolean indexing mask is created by the param_... list
            # that contains the active parameter values for each tested configuration
            # example:
            # mean_test_score = [0.5, 0.4, 0.3, 0.4] indicates four tested parameter configs
            # and param_num_layers = [1, 1, 2, 2] indicates the active values for num_layers in each tested config
            # for value=1 we get the mask [True, True, False, False], so the stored scores are [0.5, 0.4]
            # for more details on cv_results_ refer to:
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            scores = results['mean_test_score'][results[f'param_{param}'] == value]
            param_name = param.replace('kerasclassifier__', '')
            summary.append((f'{param_name}_{value}', np.mean(scores)))

    summary_df = pd.DataFrame(summary, columns=['Parameter', 'Mean Score'])
    summary_df.to_csv(output, index=False)


train_data, test_data, train_labels, test_labels, enc, smote_os, us = preprocessing.preprocess_data("./Synthetic_data.csv")
train_data_lime = train_data.copy()
train_data_lime = train_data_lime.to_numpy()

################################################
# MLP testing
################################################

base_mlp = kerasmlp.get_keras_model()
smote_pipeline = make_pipeline(smote_os, enc, base_mlp)
# randomos_pipeline = make_pipeline(random_os, enc, base_mlp)

print("Starting SMOTE MLP Training...")
grid_mlp_smote = kerasmlp.fit(smote_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_mlp_smote.best_params_}")
# add validation set before the final refit, change early stopping to monitor val_loss now
grid_mlp_smote.best_estimator_.named_steps["kerasclassifier"].set_params(validation_split=0.2,
                                                                         callbacks=EarlyStopping(
                                                                             monitor='val_loss',
                                                                             patience=10,
                                                                             min_delta=0.0001,
                                                                             restore_best_weights=True)
                                                                         )
grid_mlp_smote.best_estimator_.fit(train_data, train_labels)
mean_scores_cv(grid_mlp_smote, grid_mlp_smote.cv_results_, './output/mlp_cv_summary.csv')
history = grid_mlp_smote.best_estimator_.named_steps["kerasclassifier"].history_
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('./output/loss_plot.png')
plt.close()
predictions_mlp = grid_mlp_smote.predict(test_data)
print("SMOTE results:")
metrics.score(test_labels, predictions_mlp)

# print("Starting RandomOS MLP Training")
# grid_mlp_randomos = mlp.fit(randomos_pipeline, train_data, train_labels)
# predictions_mlp = grid_mlp_randomos.predict(test_data)
# print("RandomOS results:")
# metrics.score(test_labels, predictions_mlp)

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

base_dt = DecisionTreeClassifier(random_state=42, class_weight=None)
base_dt_weighed = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_pipeline = make_pipeline(smote_os, enc, base_dt)
dt_pipeline_without_sampling = make_pipeline(enc, base_dt_weighed)

print("Starting DT Training with Oversampling...")
grid_dt = dt.fit(dt_pipeline, train_data, train_labels)
print(f"Training finished, best params: {grid_dt.best_params_}")
mean_scores_cv(grid_dt, grid_dt.cv_results_, './output/dt_cv_summary_oversampling.csv')
predictions_dt = grid_dt.predict(test_data)
print("DT results with Oversampling:")
metrics.score(test_labels, predictions_dt)

print("Starting DT Training with weighted classes...")
grid_dt2 = dt.fit(dt_pipeline_without_sampling, train_data, train_labels)
print(f"Training finished, best params: {grid_dt2.best_params_}")
mean_scores_cv(grid_dt2, grid_dt2.cv_results_, './output/dt_cv_summary_weighted.csv')
predictions_dt2 = grid_dt2.predict(test_data)
print("DT results with weighted classes:")
metrics.score(test_labels, predictions_dt2)
