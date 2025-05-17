import json
import os
import pickle
import shutil

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from imblearn.pipeline import make_pipeline
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier

import dt
import hkb
import kerasmlp
import metrics
import preprocessing

os.makedirs('./output', exist_ok=True)

train_data, test_data, train_labels, test_labels, enc, dt_enc, smote_enn, smote_tomek, ros, tomek, smote_os, seed = preprocessing.preprocess_data("./Synthetic_data.csv")

base_mlp = kerasmlp.get_keras_model()
base_mlp.set_params(
    activation='relu',
    batch_size=64,
    epochs=100,
    l2=0.0001,
    num_layers=2,
    optimizer__learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=100, decay_rate=0.8),
    units=64,
    )
mlp_pipeline = make_pipeline(clone(smote_tomek), clone(enc), clone(base_mlp))

base_dt = DecisionTreeClassifier(
    max_features='log2',
    max_leaf_nodes=100,
    min_samples_leaf=2,
    min_samples_split=5
    )
dt_pipeline = make_pipeline(clone(smote_tomek), clone(dt_enc), clone(base_dt))

print("Starting MLP Training...")
mlp_pipeline.fit(train_data, train_labels)
train_preds_mlp = mlp_pipeline.predict(train_data)
test_preds_mlp = mlp_pipeline.predict(test_data)
print("MLP training results:")
metrics.score(train_labels, train_preds_mlp)
print("MLP results:")
metrics.score(test_labels, test_preds_mlp)
with open('./output/mlp.pkl', 'wb') as f:
    pickle.dump(mlp_pipeline, f)

print("Starting DT Training...")
dt_pipeline.fit(train_data, train_labels)
train_preds_dt = dt_pipeline.predict(train_data)
test_preds_dt = dt_pipeline.predict(test_data)
print("DT training results:")
metrics.score(train_labels, train_preds_dt)
print("DT results:")
metrics.score(test_labels, test_preds_dt)
dt.plot_dt_without_grid(dt_pipeline, "./output/dt.png")
with open('./output/dt.pkl', 'wb') as f:
    pickle.dump(dt_pipeline, f)

hkb.fit(train_data, train_labels, "2}Alter_jung,Alter_mittel,Alter_alt", "hkb.kb", train_data.shape[1])
train_preds_hkb = hkb.predict(train_data, "hkb.kb", "./output/train_preds_hkb.txt")
test_preds_hkb = hkb.predict(test_data, "hkb.kb", './output/test_preds_hkb.txt')
print("HKB training results:")
metrics.score(train_labels, train_preds_hkb)
print("HKB results:")
metrics.score(test_labels, test_preds_hkb)
shutil.copy2("hkb.kb", "./output/hkb.kb")
shutil.copy2("hkb.map", "./output/hkb.map")
shutil.copy2("hkb_discretized.sa", "./output/hkb_discretized.sa")

original_class_counts = train_labels.value_counts().to_json()
temp_resampler = clone(smote_tomek)
train_data_resampled, train_labels_resampled = temp_resampler.fit_resample(train_data, train_labels)
resampled_class_counts = train_labels_resampled.value_counts().to_json()


# logging stuff
def save_confusion_matrix(y_true, y_pred, model_name, output_path):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=['Kein','PsA','RA','SpA'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kein','PsA','RA','SpA'])
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{output_path}/{model_name}_confusion_matrix.png')
    plt.close()


save_confusion_matrix(test_labels, test_preds_mlp, "MLP", "./output")
save_confusion_matrix(test_labels, test_preds_dt, "DT", "./output")
save_confusion_matrix(test_labels, test_preds_hkb, "HKB", "./output")

np.save('./output/test_labels.npy', test_labels)
np.save('./output/test_preds_mlp.npy', test_preds_mlp)
np.save('./output/test_preds_dt.npy', test_preds_dt)
np.save('./output/test_preds_hkb.npy', test_preds_hkb)

experiment_info = {
    'random_state': seed,
    'train_shape': train_data.shape,
    'test_shape': test_data.shape,
    'original_class_counts': original_class_counts,
    'resampled_class_counts': resampled_class_counts
}

with open('./output/experiment_info.json', 'w') as f:
    json.dump(experiment_info, f, indent=4)
