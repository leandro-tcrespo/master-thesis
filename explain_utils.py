import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap.plots
import matplotlib.patches as mpatches
from lime import lime_tabular
from matplotlib.offsetbox import AnchoredText

import dt
import hkb
import metrics
import utils


def fi_explain(
    model,
    explain_data,
    background_data,
    feature_names,
    baseline,
    seed,
    plot_explanations=False,
    kb=None,
    name=None, # has to be hkb_explain, dt_explain, mlp_explain
    discretized_data_path=None,
    formatted_samples_path=None,
    pred_out=None
):
    if model == "hkb":
        predict_proba = PredictProbaWrapper(model, feature_names, name, kb, discretized_data_path,
                                            formatted_samples_path, pred_out)
        class_names = hkb.CLASS_ORDER
    else:
        predict_proba = PredictProbaWrapper(model, feature_names)
        class_names = model.steps[-1][1].classes_

    shap_explainer = shap.PermutationExplainer(predict_proba,
                                               background_data,
                                               feature_names=feature_names,
                                               seed=seed,
                                               max_evals=(2*len(feature_names)+1)*10,
                                               output_names=class_names
                                               )

    lime_explainer = lime_tabular.LimeTabularExplainer(training_data=background_data.to_numpy(),
                                                       feature_names=feature_names,
                                                       categorical_features=[i for i, col in enumerate(
                                                           background_data.columns) if col != "age"],
                                                       class_names=class_names,
                                                       random_state=seed
                                                       )

    # Get predictions and explanations, predictions are logged
    pred_inds, pred_strings = utils.get_predicted_labels(model, explain_data, name, kb,
                                                         f"./explain_data_formatted_samples.txt",
                                                         f"./output/{name}/explain_data_predictions.txt")

    shap_explanations = get_shap_explanations(shap_explainer, explain_data, pred_inds)
    lime_explanations = get_lime_explanations(lime_explainer, explain_data, predict_proba, pred_inds)

    # attribution arrays have the shape (samples, features), one attribution array per sample with an attribution for
    # each feature
    shap_attribution_arrays = get_shap_attributions(shap_explanations)
    lime_attribution_arrays = get_lime_attributions(lime_explanations, pred_inds)

    if plot_explanations:
        model_type = name.split("_")[0]
        plot_lime_explanations(lime_explanations,
                               pred_inds,
                               f"./output/{name}/lime/plots/lime_plot_{model_type}_")
        plot_shap_explanations(shap_explanations,
                               pred_strings,
                               f"./output/{name}/shap/plots/shap_plot_{model_type}_")

    metric_results_shap = metrics.score_fi_exp(model, explain_data, shap_attribution_arrays, feature_names, baseline,
                                               pred_inds, seed, kb, name, discretized_data_path,
                                               formatted_samples_path, pred_out)
    metric_results_lime = metrics.score_fi_exp(model, explain_data, lime_attribution_arrays, feature_names, baseline,
                                               pred_inds, seed, kb, name, discretized_data_path,
                                               formatted_samples_path, pred_out)

    np.savetxt(f"./output/{name}/pred_strings.txt", pred_strings, fmt="%s")

    return metric_results_shap, metric_results_lime, shap_attribution_arrays, lime_attribution_arrays


def model_explain(model, explain_data, name):
    if isinstance(model, str):
        metric_results, feature_counts = metrics.score_model_exp(model, explain_data)
        metrics.plot_feature_frequencies(feature_counts, f"./output/{name}/model_exp/plots/hkb_feature_freq_plot.png")
        rules = hkb.get_rules(explain_data, "", model, "temp_formatted_samples.txt", "temp_preds.txt")
        hkb.data_to_txt(rules, f"./output/{name}/model_exp/used_rules.txt")
    else:
        transformer = model["columntransformer"]
        dtclassifier = model["decisiontreeclassifier"]
        feature_names = transformer.get_feature_names_out()
        explain_data_transformed = transformer.transform(explain_data)
        metric_results, feature_counts = metrics.score_model_exp(dtclassifier, explain_data_transformed, feature_names)
        metrics.plot_feature_frequencies(feature_counts, f"./output/{name}/model_exp/plots/dt_feature_freq_plot.png")
        dt.plot_tree_path(model, explain_data_transformed, feature_names, f"./output/{name}/model_exp/plots/exp_dt_")
    return metric_results


def plot_lime_explanations(explanations, pred_inds, name):
    for i, explanation in enumerate(explanations):
        explanation.as_pyplot_figure(label=pred_inds[i])
        green_patch = mpatches.Patch(color='green', label='Supports Prediction')
        red_patch = mpatches.Patch(color='red', label='Contradicts Prediction')
        plt.legend(handles=[green_patch, red_patch], loc='lower left', bbox_to_anchor=(1.02, 0))

        predicted_class_prob = explanation.predict_proba[pred_inds[i]]
        lime_prediction = explanation.local_pred[0]
        intercept = explanation.intercept[pred_inds[i]]

        proba_text = f'Model Prediction: {predicted_class_prob:.3f}\nLIME Prediction: {lime_prediction:.3f}\nIntercept: {intercept:.3f}'
        prob_box = AnchoredText(proba_text,
                                loc='lower left',
                                bbox_to_anchor=(1.04, 0.15),
                                prop=dict(size=8, ha='left'),
                                frameon=True,
                                bbox_transform=plt.gca().transAxes)
        prob_box.patch.set_boxstyle("round,pad=0.4")
        plt.gca().add_artist(prob_box)
        plt.title(f"Explanation for prediction: {explanation.class_names[pred_inds[i]]}")
        plt.tight_layout()
        plt.savefig(f"{name}{i}.png")
        plt.close()


def plot_shap_explanations(explanations, pred_strings, name):
    for i, explanation in enumerate(explanations):
        shap.plots.waterfall(explanation, show=False, max_display=19)
        plt.title(f"Explanation for prediction: {pred_strings[i]}")
        plt.savefig(f"{name}{i}.png")
        plt.close()


# explain_instance calcs the feature attributions for each sample and its predicted class
def get_lime_explanations(explainer, explain_data, predict_proba, pred_inds):
    explain_data_arr = explain_data.to_numpy()
    explanations = []
    for sample, data_row in enumerate(explain_data_arr):
        explanation = explainer.explain_instance(data_row=data_row,
                                                 predict_fn=predict_proba,
                                                 labels=[pred_inds[sample]],
                                                 num_features=len(data_row))
        explanations.append(explanation)
    return explanations


# explainer(data) calcs the feature attributions for all samples and returns an explanation object,
# .values contains the attributions for all samples and classes, shape is (num_samples, num_features, num_classes)
# individual explanations for each sample and their predicted class are stored in selected_shap_explanations,
# they are accessed by essentially slicing the explanation object with all samples,
# thus by slicing the shape (num_samples, num_features, num_classes) with the corresponding sample and class we are
# interested in we get an explanation object that contains its attributions in .values with shape (num_features,),
# every individual explanation also contains its corresponding data row, so it must not be saved (data privacy),
# .values can be saved though
def get_shap_explanations(explainer, explain_data, pred_inds):
    all_shap_explanations = explainer(explain_data)
    selected_shap_explanations = []
    for sample, pred_ind in enumerate(pred_inds):
        selected_shap_explanations.append(all_shap_explanations[sample, :, pred_ind])
    return selected_shap_explanations


def get_shap_attributions(explanations):
    return [explanation.values for explanation in explanations]


# get attributions from lime in order of features in the data
# local_exp is a list of tuples that contains (index of feature in data, attribution of feature), the attributions array
# for each sample is constructed by assigning the attribution of each feature to its corresponding feature index
def get_lime_attributions(explanations, pred_inds):
    attribution_arrays = []
    for sample, explanation in enumerate(explanations):
        attr_tuples = explanation.local_exp[pred_inds[sample]]
        num_features = len(attr_tuples)
        attributions = np.zeros(num_features)
        for feature_index, attribution in attr_tuples:
            attributions[feature_index] = attribution
        attribution_arrays.append(attributions)
    return attribution_arrays


# wraps predict_proba methods for use with shap and lime, model can be either a dt/mlp pipeline or string "hkb"
class PredictProbaWrapper:
    def __init__(self, model, feature_names, name=None, kb=None, discretized_data_path=None,
                 formatted_samples_path=None, pred_out=None):
        self.model = model
        self.feature_names = feature_names
        self.name = name
        self.kb = kb
        self.discretized_data_path = discretized_data_path
        self.formatted_samples_path=formatted_samples_path
        self.pred_out=pred_out

    def __call__(self, data):
        df = pd.DataFrame(data, columns=self.feature_names)
        if self.model == "hkb":
            # print("Input samples:", len(df))
            preds = hkb.predict_proba(df, self.name, self.kb, self.discretized_data_path, self.formatted_samples_path,
                                      self.pred_out)
            return preds
        # print("Input samples:", len(df))
        return self.model.predict_proba(df)
