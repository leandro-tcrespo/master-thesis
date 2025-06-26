import os
import re
import shlex
import subprocess

import numpy as np


# def convert_cat_features(df):
#     for col in df.columns:
#         if col == 'sex':
#             df[col] = df[col].apply(lambda val: "female" if val in [1] else "male")
#         elif col == 'early':
#             df[col] = df[col].apply(lambda val: f'{col}_{val}')
#         elif col == 'age':
#             df[col] = df[col].apply(lambda val: val)
#         else:
#             df[col] = df[col].apply(lambda val: f'{col}_{val}' if val in [1, 2] else f'{col}_missing')
#     return df


def convert_cat_features(df):
    for col in df.columns:
        if col == 'sex':
            df[col] = df[col].apply(lambda val: "female" if val in [1] else "male")
        elif col == 'age':
            df[col] = df[col].apply(lambda val: val)
        else:
            df[col] = df[col].apply(lambda val: f'{col}_{val}' if val in [1, 2] else f'{col}_missing')
    return df


# this is only relevant for properly using InteKRators infer module, not relevant for fitting model
# TODO: name might be misleading, this is more than converting num features, its converting samples that are to be predicted into a list
def convert_num_features(data):
    formatted_data = []
    for index, row in data.iterrows():
        row_list = row.astype(str)
        if "age" in data.columns:
            age_index = data.columns.get_loc('age')
            row_list.iloc[age_index] = f"S{age_index+1}:" + row_list.iloc[age_index]
        row = ' '.join(row_list)
        formatted_data.append(row)
    return formatted_data


# converts list to a text file in the right data format for InteKRator, only for debugging
def data_to_txt(formatted_data, outfile="hkb_test_samples.txt"):
    with open(outfile, 'w') as file:
        for line in formatted_data:
            file.write(line + '\n')


# this checks if samples that are passed to InteKRator are in the right format, technically this makes duplicate
# features possible (male 50 a_1 a_1 b_2 ...) but usually this should not happen, this method is mostly for making sure
# that the data passed is not totally wrong
def check_state(item, age_index):
    parts = item.split()
    for part in parts:
        if part in {"female", "male"}:
            continue
        elif age_index != -1 and re.match(rf'^S{age_index+1}+:\d+(\.\d+)?$', part):
            continue
        elif re.match(r'^[a-q]_(1|2|missing)$', part):
            continue
        else:
            return False
    return True


def fit(train_data, train_labels, cluster_size, kb, train_in="hkb_train_data.txt", preselect_value=19):
    while True:
        try:
            print(f"Fitting HKB with {preselect_value} features...")
            train_data_copy = train_data.copy()
            convert_cat_features(train_data_copy)
            train_labels_copy = train_labels.copy()
            diag_multi_col = train_labels_copy.pop("diag_multi")
            train_data_copy.insert(preselect_value, "diag_multi", diag_multi_col)
            train_data_copy.to_csv(train_in, sep=' ', index=False, header=False)
            if "age" in train_data_copy.columns:
                age_index = train_data_copy.columns.get_loc('age')
                command = ["java", "-Xmx4g", "-jar", "InteKRator.jar", "-learn", "all", "discretize", cluster_size, "info",
                           str(age_index+1), "preselect", str(preselect_value), "avoid", "_missing", train_in, kb]
            else:
                command = ["java", "-Xmx4g", "-jar", "InteKRator.jar", "-learn", "all", "preselect", str(preselect_value), "avoid", "_missing", train_in, kb]
            # clear knowledge base so failed fit is not covered up by previous successful fit this is
            # necessary since InteKRator may fail without raising a CalledProcessError and leave knowledge.kb unchanged
            with open(kb, 'w') as file:
                file.write('')
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            with open('./output/fit_output.txt', 'w') as o:
                o.write(result.stdout)
                o.write(result.stderr)
            if os.path.getsize(kb) == 0:
                raise ValueError("The knowledge base is empty, HKB fitting probably failed."
                                 "Check './output/fit_output.txt' for details.")
            print(f"HKB fitted successfully. {preselect_value} features were used for training.")
            break
        except subprocess.CalledProcessError as e:
            with open('./output/fit_output.txt', 'w') as o:
                o.write(e.stdout)
                o.write(e.stderr)
            error_message = e.stdout + e.stderr
            if "OutOfMemoryError" in error_message or "java.lang.OutOfMemoryError" in error_message:
                print(f"HKB fitting failed due to memory issues. Retrying with {preselect_value} features...")
                if preselect_value > 1:
                    preselect_value -= 1
                else:
                    print("Can't reduce amount features anymore. Giving up.")
                    raise ValueError("HKB fitting failed due to memory issues. "
                                     "Check './output/fit_output.txt' for details.")

            else:
                raise ValueError("HKB fitting failed. Check './output/fit_output.txt' for details.")


def check(kb, outfile):
    command = ["java", "-Xmx4g", "-jar", "InteKRator.jar", "-check", "details", "hkb_train_data.txt", kb, outfile]
    subprocess.run(command, capture_output=True, text=True, check=True)


def intekrator_infer(item, pred_out, kb):
    try:
        command = (["java", "-jar", "InteKRator.jar", "-infer", "why"]
                   + shlex.split(item)
                   + [kb, "inference.txt"])
        # clear inference file so failed inference is not covered up by previous successful inference, this is
        # necessary since InteKRator may fail without raising a CalledProcessError and leave inference.txt unchanged
        with open('inference.txt', 'w') as file:
            file.write('')
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        with open('./output/infer_output.txt', 'w') as o:
            o.write(result.stdout)
            o.write(result.stderr)
        with open('inference.txt', 'r') as file:
            line = file.readline().strip()
        # this case should never be reached as long as the fit was successful
        if line == "":
            raise ValueError(f"No inference possible for state \n{item}.\n"
                             f"Is the knowledge base empty or without top rule? "
                             f"If not, check './output/infer_output.txt' for details.")
        with open(pred_out, 'a') as append_file:
            # append_file.write(item + '\n')
            append_file.write(line + '\n')
        return line
    except subprocess.CalledProcessError as e:
        with open('./output/infer_output.txt', 'w') as o:
            o.write(e.stdout)
            o.write(e.stderr)
        raise ValueError(f"Inference failed for state {item}. Check './output/infer_output.txt' for details.")


def predict(data, kb, data_out="hkb_test_samples.txt", pred_out='predictions.txt'):
    print("Starting inference from HKB.")
    with open(pred_out, 'w') as file:
        file.write('')
    data_copy = data.copy()
    predictions = np.array([])
    convert_cat_features(data_copy)
    age_index = -1
    if "age" in data.columns:
        age_index = data.columns.get_loc("age")
    formatted_data = convert_num_features(data_copy)
    data_to_txt(formatted_data, data_out)
    for item in formatted_data:
        if check_state(item, age_index):
            line = intekrator_infer(item, pred_out, kb)
            prediction = line.split('   (')[0]
            predictions = np.append(predictions, prediction)
            with open(pred_out, 'a') as append_file:
                append_file.write('\n')
        else:
            raise ValueError(f"Invalid format for state {item}.")
    print(f"Inference successful. Check {pred_out} for all predictions.")
    return predictions


def predict_proba(data, pred_out='predictions.txt'):
    print("Starting inference from HKB.")
    with open(pred_out, 'w') as file:
        file.write('')
    data_copy = data.copy()
    convert_cat_features(data_copy)
    formatted_data = convert_num_features(data_copy)
    data_to_txt(formatted_data)
    predictions = np.empty((0, 4))
    for item in formatted_data:
        if check_state(item):
            line = intekrator_infer(item, pred_out)
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
            elif prediction == "PsA":
                pred_probas = np.array([other_proba, other_proba, other_proba, pred_proba])
            else:
                raise ValueError(f"Predicted label {prediction} is not valid.")
            with open(pred_out, 'a') as append_file:
                append_file.write(str(pred_probas) + '\n')
                append_file.write('\n')
            predictions = np.append(predictions, [pred_probas], axis=0)
        else:
            raise ValueError(f"Invalid format for state {item}.")
    print(f"Inference successful. Check {pred_out} for all predictions.")
    return predictions
