import os
import re
import subprocess

import numpy as np

CLASS_ORDER = ['Kein', 'RA', 'SpA', 'PsA']

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


# returns the rule a prediction is based on (first rule if multiple rules are possible)
def extract_rule(line):
    start = line.find(":")
    end = line.rfind(")")
    all_rules = line[start+1:end]
    rule = all_rules.split(",")[0].strip()
    return rule


# tokenizes the rule into a list of states, the prediction and the conditional probability tied to the conclusion,
# assumes that the -why option is used when inferring with InteKRator
def tokenize_rule(rule):
    states = []
    if "->" in rule:
        premise, conclusion = [part.strip() for part in rule.split("->")]
        states = [state.strip() for state in premise.split("^")]
    else:
        conclusion = rule.strip()
    prediction, after_pred = [part.strip() for part in conclusion.split("[")]
    cond_proba = after_pred.split("]")[0].strip()
    return states, prediction, cond_proba


def convert_cat_features(df):
    for col in df.columns:
        if col == 'sex':
            df[col] = df[col].apply(lambda val: "female" if val in [1] else "male")
        elif col == 'age':
            df[col] = df[col].apply(lambda val: val)
        else:
            df[col] = df[col].apply(lambda val: f'{col}_{int(val)}' if val in [1, 2] else f'{col}_missing')
    return df


# this is only relevant for properly using InteKRators infer module, not relevant for fitting model
# TODO: name might be misleading, this is more than converting num features, its converting samples that are to be predicted into a list
def convert_num_features(data):
    formatted_data = []
    age_index = -1
    for index, row in data.iterrows():
        row_list = row.astype(str)
        if "age" in data.columns:
            age_index = data.columns.get_loc('age')
            row_list.iloc[age_index] = f"S{age_index+1}:" + row_list.iloc[age_index]
        row = ' '.join(row_list)
        if check_state(row, age_index):
            formatted_data.append(row)
        else:
            raise ValueError(f"Invalid format for state {row}.")
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


def fit(train_data, train_labels, name, cluster_size, kb, train_in="hkb_train_data.txt", preselect_value=19):
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
            # clear kb file so result of fit is checked to catch problems that do not raise an error
            # (for example InteKRator may print out errors into the console but not raise an error)
            with open(kb, 'w') as file:
                file.write('')
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            with open(f'./output/{name}/fit_output.txt', 'w') as o:
                o.write(result.stdout)
                o.write(result.stderr)
            if os.path.getsize(kb) == 0:
                raise ValueError("The knowledge base is empty, HKB fitting probably failed."
                                 f"Check './output/{name}/fit_output.txt' for details.")
            print(f"HKB fitted successfully. {preselect_value} features were used for training.")
            break
        except subprocess.CalledProcessError as e:
            with open(f'./output/{name}/fit_output.txt', 'w') as o:
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
                                     f"Check './output/{name}/fit_output.txt' for details.")

            else:
                raise ValueError(f"HKB fitting failed. Check './output/{name}/fit_output.txt' for details.")


def intekrator_infer(name, kb, formatted_samples_path, pred_out):
    try:
        command = (["java", "-jar", "InteKRator.jar", "-inferMulti", "why", formatted_samples_path, kb, pred_out])
        # clear inference file so result of inferMulti is checked to catch problems that do not raise an error
        # (for example InteKRator may print out errors into the console but not raise an error)
        with open(pred_out, "w") as file:
            file.write("")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        with open(f'./output/{name}/infer_output.txt', 'w') as o:
            o.write(result.stdout)
            o.write(result.stderr)
        # checks if inferences were made (see above)
        if os.path.getsize(pred_out) == 0:
            raise ValueError(f"Inference failed.\n"
                             f"Is the knowledge base empty or without top rule? "
                             f"If not, check './output/{name}/infer_output.txt' for details.")
    except subprocess.CalledProcessError as e:
        with open(f'./output/{name}/infer_output.txt', 'w') as o:
            o.write(e.stdout)
            o.write(e.stderr)
        raise ValueError(f"Inference failed. Check './output/{name}/infer_output.txt' for details.")


def predict(data, name, kb, formatted_samples_path, pred_out):
    print("Starting inference from HKB.")
    data_copy = data.copy()
    convert_cat_features(data_copy)
    formatted_data = convert_num_features(data_copy)
    data_to_txt(formatted_data, formatted_samples_path)
    intekrator_infer(name, kb, formatted_samples_path, pred_out)
    predictions = []
    with open(pred_out, "r") as file:
        for line in file:
            prediction = line.split('(')[0].strip()
            if prediction:  # ignore blank lines from intekrator inferMulti output file
                predictions.append(prediction)
    print("Inference successful.")
    return np.array(predictions)


# computes support and confidence of a rule given a dataset data, the states of the premise of a rule and the
# pred from the conclusion of the rule, data must be the discretized version (.sa file) for correct matching of age
def compute_support_confidence(data, states, prediction):
    match_premise_count = 0
    match_premise_and_class_count = 0
    total_samples = len(data)

    for sample in data:
        sample_set = set(sample.split())
        states_set = set(states)
        if states_set.issubset(sample_set):
            match_premise_count += 1
            if prediction in sample_set:
                match_premise_and_class_count += 1

    support = match_premise_and_class_count / total_samples
    confidence = match_premise_and_class_count / (match_premise_count + 1e-10)

    return support, confidence


# data = ["female Alter_alt b_1 c_1 d_missing e_missing f_missing g_missing i_1 j_2 k_2 m_2 n_1 p_1 q_1 Kein", "male Alter_mittel b_1 c_1 d_missing e_missing f_missing g_2 i_1 j_2 k_2 m_1 n_2 p_1 q_2 Kein",
#         "female Alter_alt b_1 c_2 d_missing e_missing f_2 g_1 i_1 j_2 k_2 m_1 n_1 p_1 q_2 Kein",
#         "male Alter_mittel b_1 c_2 d_missing e_missing f_missing g_missing i_1 j_1 k_2 m_1 n_2 p_2 q_2 Kein",
#         "female Alter_alt b_1 c_2 d_missing e_missing f_missing g_missing i_1 j_2 k_1 m_2 n_1 p_missing q_2 Kein",
#         "male Alter_mittel b_1 c_2 d_missing e_1 f_missing g_missing i_1 j_2 k_2 m_1 n_missing p_2 q_2 Kein",
#         "female Alter_alt b_1 c_1 d_missing e_missing f_missing g_missing i_1 j_2 k_1 m_2 n_1 p_missing q_2 Kein"]
# states = "Alter_alt b_1 c_1".split()
# prediction = "Kein"
# support, confidence = compute_support(data, states, prediction)


# discretized_data_path must lead to the .sa file created during fitting of hkb, the .sa file contains the data used
# for hkb training with discretized values and is needed for calculating support and confidence to get pred_probas
def predict_proba(data, name, kb, discretized_data_path, formatted_samples_path, pred_out):
    data_copy = data.copy()
    convert_cat_features(data_copy)
    formatted_data = convert_num_features(data_copy)
    data_to_txt(formatted_data, formatted_samples_path)
    intekrator_infer(name, kb, formatted_samples_path, pred_out)
    with open(discretized_data_path, "r") as f:
        discretized_data = f.readlines()
    all_probas = []
    with open(pred_out, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                rule = extract_rule(line)
                states, prediction, cond_proba = tokenize_rule(rule)
                proba_vector = []
                for cls in CLASS_ORDER:
                    support, confidence = compute_support_confidence(discretized_data, states, cls)
                    proba_vector.append(confidence*support)
                    # print("Conf:", confidence)
                    # print("Supp:", support)

                normalized_proba_vector = [proba / sum(proba_vector) for proba in proba_vector]
                all_probas.append(normalized_proba_vector)
    return np.array(all_probas)
