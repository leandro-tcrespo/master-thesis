import json
import os
import pickle
import shutil

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import utils

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

# os.makedirs("./output", exist_ok=True)
os.makedirs("./output/mlps", exist_ok=True)
os.makedirs("./output/dts", exist_ok=True)
os.makedirs("./output/hkbs", exist_ok=True)


# logging stuff
def save_confusion_matrix(y_true, y_pred, model_name, output_path):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=['Kein','PsA','RA','SpA'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kein','PsA','RA','SpA'])
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{output_path}/{model_name}_confusion_matrix.png')
    plt.close()


def log_results(train_data, test_data, seed, id, enc, ros, tomek):
    original_class_counts = train_labels.value_counts().to_json()
    train_data_resampled, train_labels_resampled = utils.resample_data(enc, train_data, train_labels, ros, tomek)
    resampled_class_counts = utils.count_labels(train_labels_resampled).to_json()

    experiment_info = {
        'random_state': seed,
        'train_shape': train_data.shape,
        'test_shape': test_data.shape,
        'original_class_counts': original_class_counts,
        'resampled_class_counts': resampled_class_counts
    }

    with open(f"./output/experiment_info_{id}.json", "w") as f:
        json.dump(experiment_info, f, indent=4)


mlp_f = open("./output/mlp_results.txt", "a")
dt_f = open("./output/dt_results.txt", "a")
hkb_f = open("./output/hkb_results.txt", "a")


def fit_and_test_mlp(train_data, test_data, train_labels, test_labels, enc, ros, tomek, seed, id):
    base_mlp = kerasmlp.get_keras_model()
    base_mlp.set_params(
        activation='relu',
        batch_size=64,
        epochs=150,
        l2=0.0001,
        num_layers=2,
        optimizer__learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=50, decay_rate=0.9),
        units=64,
        random_state=seed,
        )
    mlp_pipeline = make_pipeline(clone(ros), clone(enc), clone(tomek), clone(base_mlp))

    mlp_pipeline.fit(train_data, train_labels)
    train_preds_mlp = mlp_pipeline.predict(train_data)
    test_preds_mlp = mlp_pipeline.predict(test_data)
    metrics.score(train_labels, train_preds_mlp, f"MLP_{id} training results:", mlp_f)
    metrics.score(test_labels, test_preds_mlp, f"MLP_{id} results:", mlp_f)
    with open(f"./output/mlps/mlp_{id}.pkl", "wb") as f:
        pickle.dump(mlp_pipeline, f)
    return test_preds_mlp


def fit_and_test_dt(train_data, test_data, train_labels, test_labels, enc, ros, tomek, seed, id):
    base_dt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=seed,
        )
    dt_pipeline = make_pipeline(clone(ros), clone(enc), clone(tomek), clone(base_dt))

    dt_pipeline.fit(train_data, train_labels)
    train_preds_dt = dt_pipeline.predict(train_data)
    test_preds_dt = dt_pipeline.predict(test_data)
    metrics.score(train_labels, train_preds_dt, f"DT_{id} training results:", dt_f)
    metrics.score(test_labels, test_preds_dt, f"DT_{id} results:", dt_f)
    dt.plot_dt_without_grid(dt_pipeline, f"./output/dts/dt_{id}.png")
    with open(f"./output/dts/dt_{id}.pkl", "wb") as f:
        pickle.dump(dt_pipeline, f)
    return test_preds_dt


def fit_and_test_hkb(train_data, test_data, train_labels, test_labels, id):
    if "age" in train_data.columns:
        age_index = train_data.columns.get_loc('age')
        hkb.fit(train_data, train_labels, f"{age_index+1}}}Alter_jung,Alter_mittel,Alter_alt", f"hkb_{id}.kb", f"hkb_train_data_{id}.txt", train_data.shape[1])
    else:
        hkb.fit(train_data, train_labels, "", f"hkb_{id}.kb", f"hkb_train_data_{id}.txt", train_data.shape[1])
    train_preds_hkb = hkb.predict(train_data, f"hkb_{id}.kb", f"hkb_train_samples_{id}.txt", f"./output/hkbs/train_preds_hkb_{id}.txt")
    test_preds_hkb = hkb.predict(test_data, f"hkb_{id}.kb", f"hkb_test_samples_{id}.txt", f'./output/hkbs/test_preds_hkb_{id}.txt')
    metrics.score(train_labels, train_preds_hkb, f"HKB_{id} training results:", hkb_f)
    metrics.score(test_labels, test_preds_hkb, f"HKB_{id} results:", hkb_f)
    shutil.copy2(f"hkb_{id}.kb", f"./output/hkbs/hkb_{id}.kb")
    if "age" in train_data.columns:
        shutil.copy2(f"hkb_{id}.map", f"./output/hkbs/hkb_{id}.map")
    return test_preds_hkb


for i in range(5):
    (train_data, test_data, train_labels, test_labels, enc,
     ros, tomek, smote, seed, enn) = (preprocessing.preprocess_data("./Synthetic_data.csv"))
    # edit seeds for each iteration here if needed
    seeds = [seed]*5
    seed = seeds[i]
    print("Starting MLP Training...")
    test_preds_mlp = fit_and_test_mlp(train_data, test_data, train_labels, test_labels, enc, ros, tomek, seed, i)
    print("Starting DT Training...")
    test_preds_dt = fit_and_test_dt(train_data, test_data, train_labels, test_labels, enc, ros, tomek, seed, i)
    print("Starting HKB Training...")
    test_preds_hkb = fit_and_test_hkb(train_data, test_data, train_labels, test_labels, i)

    save_confusion_matrix(test_labels, test_preds_mlp, f"MLP_{i}", "./output/mlps")
    save_confusion_matrix(test_labels, test_preds_dt, f"DT_{i}", "./output/dts")
    save_confusion_matrix(test_labels, test_preds_hkb, f"HKB_{i}", "./output/hkbs")

    np.save(f"./output/test_labels_{i}.npy", test_labels)
    np.save(f"./output/mlps/test_preds_mlp_{i}.npy", test_preds_mlp)
    np.save(f"./output/dts/test_preds_dt_{i}.npy", test_preds_dt)
    np.save(f"./output/hkbs/test_preds_hkb_{i}.npy", test_preds_hkb)

    log_results(train_data, test_data, seed, i, enc, ros, tomek)

mlp_f.close()
dt_f.close()
hkb_f.close()
