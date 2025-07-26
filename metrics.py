import json

import numpy as np
from imblearn.metrics import specificity_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import pandas as pd

import hkb


def score(y_true, y_pred, modelname, filename):

    label_order = ['Kein','PsA','RA','SpA']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    recall = recall_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    spec = specificity_score(y_true, y_pred, labels=label_order, average=None)

    print(modelname, file=filename)
    print("Class     Precision   Recall      F1-Score    Specificity", file=filename)
    print("---------------------------------------------------------", file=filename)
    print(f"Kein      {precision[0]:.2f}        {recall[0]:.2f}        {f1[0]:.2f}        {spec[0]:.2f}", file=filename)
    print(f"PsA       {precision[1]:.2f}        {recall[1]:.2f}        {f1[1]:.2f}        {spec[1]:.2f}", file=filename)
    print(f"RA        {precision[2]:.2f}        {recall[2]:.2f}        {f1[2]:.2f}        {spec[2]:.2f}", file=filename)
    print(f"SpA       {precision[3]:.2f}        {recall[3]:.2f}        {f1[3]:.2f}        {spec[3]:.2f}", file=filename)
    print("---------------------------------------------------------", file=filename)
    print(f"Avg       {precision.mean():.2f}        {recall.mean():.2f}        {f1.mean():.2f}        {spec.mean():.2f}", file=filename)
    print("---------------------------------------------------------", file=filename)
    print(f"Accuracy  {accuracy}", file=filename)
    print("---------------------------------------------------------", file=filename)


# implementation of complexity measure as defined in "Evaluating and Aggregating Feature-based Model Explanations" -
# https://arxiv.org/abs/2005.00631
# implementation similar to BEExAI implementation - https://github.com/SquareResearchCenter-AI/BEExAI/tree/main
def compute_complexity(attributions):
    abs_attributions = np.abs(attributions)
    P_g = abs_attributions / (np.sum(abs_attributions) + 1e-10)

    complexity = -np.sum(P_g * np.log(P_g + 1e-10))

    return complexity


def avg_complexities(all_attributions):
    complexities = [compute_complexity(attributions) for attributions in all_attributions]
    mean_complexity = np.mean(complexities) if complexities else 0
    return complexities, mean_complexity


# implementation of faithfulness correlation as defined in
# "Evaluating and Aggregating Feature-based Model Explanations" - https://arxiv.org/abs/2005.00631
# implementation similar to BEExAI implementation - https://github.com/SquareResearchCenter-AI/BEExAI/tree/main
def faithfulness_corr(model, input_sample, attributions, feature_names, baseline, label, name, kb,
                      discretized_data_path, formatted_samples_path, pred_out, n_repeats=20,
                      n_features_subset=3,):
    # floats, so the mean value for age from the baseline is used when replacing indices for perturbed array
    input_sample = np.array(input_sample).astype(float)
    baseline = np.array(baseline)

    input_df = pd.DataFrame([input_sample], columns=feature_names)
    if model == "hkb":
        original_probs = hkb.predict_proba(input_df, name, kb, discretized_data_path, formatted_samples_path,
                                           pred_out).flatten()
    else:
        original_probs = model.predict_proba(input_df).flatten()
    original_pred = original_probs[label]

    prediction_diffs = []
    attribution_sums = []

    for i in range(n_repeats):
        replace_indices = np.random.choice(len(input_sample), n_features_subset, replace=False)

        perturbed = input_sample.copy()
        perturbed[replace_indices] = baseline[replace_indices]

        perturbed_df = pd.DataFrame([perturbed], columns=feature_names)
        if model == "hkb":
            perturbed_probs = hkb.predict_proba(perturbed_df, name, kb, discretized_data_path, formatted_samples_path,
                                           pred_out).flatten()
        else:
            perturbed_probs = model.predict_proba(perturbed_df).flatten()
        perturbed_pred = perturbed_probs[label]

        delta = original_pred - perturbed_pred
        attr_sum = attributions[replace_indices].sum()

        prediction_diffs.append(delta)
        attribution_sums.append(attr_sum)

    prediction_diffs = np.array(prediction_diffs)
    attribution_sums = np.array(attribution_sums)

    if np.all(prediction_diffs == prediction_diffs[0]) or np.all(attribution_sums == attribution_sums[0]):
        return np.nan
    return np.corrcoef(prediction_diffs, attribution_sums)[0, 1]


def avg_faithfulness_corr(model, data, all_attributions, feature_names, baseline, labels, seed, name, kb,
                          discretized_data_path, formatted_samples_path, pred_out):
    np.random.seed(seed)
    data = np.array(data)
    faithfulness_scores = []
    for i, sample in enumerate(data):
        faithfulness_scores.append(
            faithfulness_corr(model, sample, all_attributions[i], feature_names, baseline, labels[i], name,
                              kb, discretized_data_path, formatted_samples_path, pred_out))
    if np.nan in faithfulness_scores:
        print("Some faithfulness_scores are nan. Ignoring nan values for mean score.")
    return faithfulness_scores, np.nanmean(faithfulness_scores)


def avg_explanation_length(model, explain_data):
    if isinstance(model, str):
        premises = hkb.get_premises(explain_data, "", model, "temp_formatted_samples.txt", "temp_preds.txt")
        premise_lengths = []
        for premise in premises:
            premise_lengths.append(len(premise))
        return np.mean(premise_lengths)
    else:
        # returns (samples, nodes) sparse matrix where for each sample i and node j (i,j)=1 if sample passed through
        # node and (i,j)=0 if sample did not
        decision_paths = model.decision_path(explain_data)
        # sums up all nodes each sample passes through (sums up all ones), axis=1 because nodes are the columns
        path_lengths = decision_paths.sum(axis=1).flatten()
        # remove leaf node to only consider decision splits
        return np.mean(path_lengths-1)


def count_features(model, explain_data, feature_names=None):
    feature_counts = Counter()
    if isinstance(model, str):
        premises = hkb.get_premises(explain_data, "", model, "temp_formatted_samples.txt", "temp_preds.txt")
        for premise in premises:
            for feature in premise:
                feature_counts[feature] += 1
    else:
        paths = model.decision_path(explain_data)
        features = model.tree_.feature

        for path in paths:
            path_nodes = path.nonzero()
            path_nodes = path_nodes[1]
            for node in path_nodes:
                feature_id = features[node]
                if feature_id >= 0:
                    feature_counts[feature_names[feature_id]] += 1

    return feature_counts


def plot_normalized_frequencies(counter, name):
    total = sum(counter.values())
    normalized = {k: v / total for k, v in counter.items()}

    features = list(normalized.keys())
    frequencies = list(normalized.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, frequencies)
    plt.xlabel('Features')
    plt.ylabel('Normalized Frequency')
    plt.title('Normalized Feature Usage Frequencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()
    return

# calcs baseline for faithfulness correlation by taking modes for categorical features and mean for age feature
def get_baseline(df):
    # get first row in case some modes are tied for some features
    feature_vector = df.mode().iloc[0]
    # after getting modes for all categorical features, overwrite value for age feature with mean
    if "age" in df.columns:
        feature_vector["age"] = df["age"].mean()
    return feature_vector.to_numpy()


# Calculate metrics
def score_fi_exp(model, explain_data, all_attributions, feature_names,
                 baseline, pred_inds, seed, kb, name, discretized_data_path, formatted_samples_path, pred_out):
    cmplx_scores, cmplx_avg = avg_complexities(all_attributions)
    fthfl_scores, fthfl_avg = avg_faithfulness_corr(model, explain_data, all_attributions, feature_names, baseline,
                                                    pred_inds, seed, name, kb, discretized_data_path,
                                                    formatted_samples_path, pred_out)
    cmplx_scores_json = json.dumps(cmplx_scores)
    fthfl_scores_json = json.dumps(fthfl_scores)
    metric_results = {
                "cmplx_scores": cmplx_scores_json,
                "cmplx_avg": cmplx_avg,
                "fthfl_scores": fthfl_scores_json,
                "fthfl_avg": fthfl_avg,
                "seed": seed,
                     }
    return metric_results


def score_model_exp(model, explain_data, feature_names=None):
    exp_length_avg = avg_explanation_length(model, explain_data)
    feature_counts = count_features(model, explain_data, feature_names)
    feature_counts_json = json.dumps(feature_counts)
    metric_results = {
        "exp_length_avg": exp_length_avg,
        "feature_counts": feature_counts_json
    }
    return metric_results, feature_counts
