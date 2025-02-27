import numpy as np
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
import pandas as pd


def score(y_true, y_pred):

    label_order = ['Kein','PsA','RA','SpA']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    recall = recall_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, labels=label_order, average=None, zero_division=0.0)
    spec = specificity_score(y_true, y_pred, labels=label_order, average=None)

    print("Class     Precision   Recall      F1-Score    Specificity")
    print("---------------------------------------------------------")
    print(f"Kein      {precision[0]:.2f}        {recall[0]:.2f}        {f1[0]:.2f}        {spec[0]:.2f}")
    print(f"PsA       {precision[1]:.2f}        {recall[1]:.2f}        {f1[1]:.2f}        {spec[1]:.2f}")
    print(f"RA        {precision[2]:.2f}        {recall[2]:.2f}        {f1[2]:.2f}        {spec[2]:.2f}")
    print(f"SpA       {precision[3]:.2f}        {recall[3]:.2f}        {f1[3]:.2f}        {spec[3]:.2f}")
    print("---------------------------------------------------------")
    print(f"Avg       {precision.mean():.2f}        {recall.mean():.2f}        {f1.mean():.2f}        {spec.mean():.2f}")
    print("---------------------------------------------------------")
    print(f"Accuracy  {accuracy}")
    print("---------------------------------------------------------")


# implementation of complexity measure as defined in "Evaluating and Aggregating Feature-based Model Explanations" -
# https://arxiv.org/abs/2005.00631
# implementation similar to BEExAI implementation - https://github.com/SquareResearchCenter-AI/BEExAI/tree/main
def compute_complexity(fi_values):
    abs_fi_values = np.abs(fi_values)
    P_g = abs_fi_values / (np.sum(abs_fi_values) + 1e-10)

    complexity = -np.sum(P_g * np.log(P_g + 1e-10))

    return complexity


# todo: basic functionality done, will have to look into what baseline is chosen ("all zeros" or one baseline value
#  per feature, rn just passing array full with 1s that doesn't work for age for example),
#  also will have to look into how baseline is correctly applied if not all features used in training
#  are used for explanations (to make explanations easier to read), in this case the non-selected features(by Shap/Lime)
#  would have to be fixed, so they are not perturbed (only the selected features should be perturbed since these are
#  deemed the most important features by Lime), it would be better to compute faithfulness_corr with all features
#  able to be perturbed but in the end its a tradeoff between interpretability and accuracy of FC
#  more features => more subsets explorable but also less readable explanation
# implementation of faithfulness correlation as defined in
# "Evaluating and Aggregating Feature-based Model Explanations" - https://arxiv.org/abs/2005.00631
# implementation similar to BEExAI implementation - https://github.com/SquareResearchCenter-AI/BEExAI/tree/main
def faithfulness_corr(
        model,
        input_sample,
        attributions,
        feature_names,
        n_repeats=20,
        n_features_subset=5,
        baseline=None,
        label=None
):
    original_probs = model.predict_proba(pd.DataFrame([input_sample],columns=feature_names))[0]

    original_pred = original_probs[label]

    prediction_diffs = []
    attribution_sums = []

    for _ in range(n_repeats):
        replace_indices = np.random.choice(len(input_sample), n_features_subset, replace=False)

        perturbed = input_sample.copy()
        perturbed[replace_indices] = baseline[replace_indices]

        perturbed_probs = model.predict_proba(pd.DataFrame([perturbed],columns=feature_names))[0]
        perturbed_pred = perturbed_probs[label]

        delta = original_pred - perturbed_pred

        attr_sum = attributions[replace_indices].sum()

        prediction_diffs.append(delta)
        attribution_sums.append(attr_sum)

    return pearsonr(prediction_diffs, attribution_sums)[0]
