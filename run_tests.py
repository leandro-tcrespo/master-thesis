import pandas as pd
from imblearn.pipeline import make_pipeline
from lime import lime_tabular
from sklearn.neural_network import MLPClassifier

import hkb
import lime_utils
import mlp
import preprocessing


def predict_proba_wrapper_mlp(original_data):
    original_data_copy = pd.DataFrame(original_data, columns=train_data.columns)
    return grid_mlp_smote.predict_proba(original_data_copy)


def predict_proba_wrapper_hkb(original_data):
    original_data_df = pd.DataFrame(original_data, columns=train_data.columns)
    non_age_cols = original_data_df.columns.difference(['age'])
    original_data_df[non_age_cols] = original_data_df[non_age_cols].astype(int)
    return hkb.predict_proba(original_data_df, 'lime_predictions_hkb')


train_data, test_data, train_labels, test_labels, enc, smote_os, random_os = preprocessing.preprocess_data("./Synthetic_data.csv")
train_data_lime = train_data.copy()
train_data_lime = train_data_lime.to_numpy()

base_mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
smote_pipeline = make_pipeline(smote_os, enc, base_mlp)
randomos_pipeline = make_pipeline(random_os, enc, base_mlp)

print("Starting SMOTE MLP Training")
grid_mlp_smote = mlp.fit(smote_pipeline, train_data, train_labels)
predictions_mlp = grid_mlp_smote.predict(test_data)
print("SMOTE results:")
mlp.score(test_labels, predictions_mlp)

print("Starting RandomOS MLP Training")
grid_mlp_randomos = mlp.fit(randomos_pipeline, train_data, train_labels)
predictions_mlp = grid_mlp_randomos.predict(test_data)
print("RandomOS results:")
mlp.score(test_labels, predictions_mlp)

# todo add categorical names to make lime explanations more understandable,
#  probably have to convert features 1,2,3,9 to 0,1,2,3 because
#  categorical names dict is accessed by looking at index indicated by feature value i in column x: names[x][i]
explainer_mlp = lime_tabular.LimeTabularExplainer(
    train_data_lime,
    categorical_features=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    feature_names=['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'],
    class_names=grid_mlp_smote.classes_.tolist(),
    verbose=True,
    random_state=42
)
exp_mlp = explainer_mlp.explain_instance(test_data.values[0], predict_proba_wrapper_mlp, top_labels=1)
fig = lime_utils.plot_lime(exp_mlp, 'lime_plot_mlp.png')
fig.show()

print("Starting HKB Training")
hkb.fit(train_data, train_labels)
predictions = hkb.predict(test_data)
print("HKB results:")
hkb.score(test_labels, predictions)

explainer_hkb = lime_tabular.LimeTabularExplainer(
    train_data_lime,
    categorical_features=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    feature_names=['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'],
    class_names=["Kein", "PsA", "RA", "SpA"],
    verbose=True,
    random_state=42
)
exp_hkb = explainer_hkb.explain_instance(test_data.values[0], predict_proba_wrapper_hkb, top_labels=1)
fig = lime_utils.plot_lime(exp_hkb, 'lime_plot_hkb.png')
fig.show()
