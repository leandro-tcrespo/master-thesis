import shlex
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def convert_to_intekrator(df):
    for col in df.columns:
        if col == 'sex':
            df[col] = df[col].apply(lambda val: "female" if val in [1] else "male")
        else:
            df[col] = df[col].apply(lambda val: f'{col}_{val}' if val in [1, 2, 3, 9] else val)
    return df


def fit(train_data, train_labels):
    train_data_copy = train_data.copy()
    convert_to_intekrator(train_data_copy)
    train_labels_copy = train_labels.copy()
    diag_multi_col = train_labels_copy.pop("diag_multi")
    train_data_copy.insert(19, "diag_multi", diag_multi_col)
    train_data_copy.to_csv("hkb_train_data.txt", sep=' ', index=False, header=False)
    command = ["java", "-jar", "InteKRator.jar", "-learn", "all", "discretize", "2}3", "info",
               "any", "hkb_train_data.txt", "knowledge.kb"]
    subprocess.run(command, capture_output=True, shell=True)


def predict(data, pred_out='predictions.txt'):
    with open(pred_out, 'w') as file:
        file.write('')
    test_data_copy = data.copy()
    convert_to_intekrator(test_data_copy)
    formatted_data = []
    predictions = np.array([])
    for index, row in test_data_copy.iterrows():
        row_list = row.astype(str)
        row_list.iloc[1] = 'S2:' + row_list.iloc[1]
        row = ' '.join(row_list)
        formatted_data.append(row)
    for item in formatted_data:
        command = (["java", "-jar", "InteKRator.jar", "-infer", "why"]
                   + shlex.split(item)
                   + ["knowledge.kb","inference.txt"])
        subprocess.run(command, capture_output=True, shell=True)
        with open('inference.txt', 'r') as file:
            line = file.readline().strip()
        with open(pred_out, 'a') as append_file:
            append_file.write(item + '\n')
            append_file.write(line + '\n')
            append_file.write('\n')
        prediction = line.split('   (')[0]
        predictions = np.append(predictions, prediction)
    return predictions


def predict_proba(data, pred_out='predictions.txt'):
    with open(pred_out, 'w') as file:
        file.write('')
    data_copy = data.copy()
    convert_to_intekrator(data_copy)
    data.to_csv("intekrator_test.txt", sep=' ', index=False, header=False)
    formatted_data = []
    predictions = np.empty((0,4))
    for index, row in data_copy.iterrows():
        row_list = row.astype(str)
        row_list.iloc[1] = 'S2:' + row_list.iloc[1]
        row = ' '.join(row_list)
        formatted_data.append(row)
    for item in formatted_data:
        command = (["java", "-jar", "InteKRator.jar", "-infer", "why"] +
                   shlex.split(item) +
                   ["knowledge.kb", "inference.txt"])
        subprocess.run(command, capture_output=True)
        with open('inference.txt', 'r') as read_file:
            line = read_file.readline().strip()
        with open(pred_out, 'a') as append_file:
            append_file.write(item + '\n')
            append_file.write(line + '\n')
            append_file.write('\n')
        if line:
            prediction = line.split('   (')[0]
            part1 = line.split('[')[1]
            pred_proba_string = part1.split(']')[0]
            pred_proba = float(pred_proba_string)
            other_proba = (1 - pred_proba)/3
            if prediction == "Kein":
                pred_probas = np.array([pred_proba, other_proba, other_proba, other_proba])
            elif prediction == "RA":
                pred_probas = np.array([other_proba, pred_proba, other_proba, other_proba])
            elif prediction == "SpA":
                pred_probas = np.array([other_proba, other_proba, pred_proba, other_proba])
            else:
                pred_probas = np.array([other_proba, other_proba, other_proba, pred_proba])
        else:
            pred_probas = np.zeros(4)
        predictions = np.append(predictions, [pred_probas], axis=0)
    return predictions


def score(y_true, y_pred):
    print("HKB Accuracy:", accuracy_score(y_true, y_pred))
    print("HKB F1:", f1_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("HKB Precision:", precision_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("HKB Recall:", recall_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("------------------")
