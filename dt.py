import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'min_impurity_decrease': [0.0, 0.01],
    'max_leaf_nodes': [10, 20, 50, 100, 200, None]
}
new_param_grid = {'decisiontreeclassifier__' + key: param_grid[key] for key in param_grid}


def fit(model, t_data, t_labels, grid):
    grid_model = GridSearchCV(model, param_grid=grid, cv=5, n_jobs=-1, scoring='f1_macro',
                              error_score='raise')

    grid_model.fit(t_data, t_labels.values.ravel())
    return grid_model


def prune(pipeline, t_data, t_labels):
    dt = pipeline.named_steps['decisiontreeclassifier']
    path = dt.cost_complexity_pruning_path(t_data, t_labels)
    ccp_grid = {'decisiontreeclassifier__ccp_alpha': path.ccp_alphas}
    grid_ccp = fit(pipeline, t_data, t_labels, ccp_grid)
    print(f"Best ccp alpha value found: {grid_ccp.best_params_}")
    print(f"All parameters of best model: {grid_ccp.best_estimator_.named_steps['decisiontreeclassifier'].get_params()}")
    return grid_ccp


def plot_dt(grid_dt, out_file):
    plt.figure(figsize=(15, 10))
    plot_tree(
        grid_dt.best_estimator_['decisiontreeclassifier'],
        feature_names=grid_dt.best_estimator_['columntransformer'].get_feature_names_out(),
        class_names=grid_dt.best_estimator_.classes_.tolist(),
        filled=True,
        rounded=True
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()


def plot_dt_without_grid(dt, out_file):
    plt.figure(figsize=(15, 10))
    plot_tree(
        dt['decisiontreeclassifier'],
        feature_names=dt['columntransformer'].get_feature_names_out(),
        class_names=dt.classes_.tolist(),
        filled=True,
        rounded=True
    )

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()


def plot_tree_path(dt, explain_data, feature_names, name):
    dtclassifier = dt["decisiontreeclassifier"]
    scaler = dt["columntransformer"].named_transformers_["AgeScaler"]
    for j, sample in enumerate(explain_data):
        node_indicator = dtclassifier.decision_path(sample.reshape(1, -1))
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        fig, ax = plt.subplots(figsize=(5, len(node_index)))
        for i, node_id in enumerate(node_index):
            left = dtclassifier.tree_.children_left[node_id]
            right = dtclassifier.tree_.children_right[node_id]
            if left == right:  # Leaf node
                label = f"Leaf: {dtclassifier.classes_[dtclassifier.tree_.value[node_id].argmax()]}"
            else:
                f_idx = dtclassifier.tree_.feature[node_id]
                feat_name = feature_names[f_idx]
                thresh = dtclassifier.tree_.threshold[node_id]
                value = sample[f_idx]

                # If the feature is scaled (example here: only 'Age'), apply inverse_transform for both value and threshold
                if feat_name == "age":
                    arr_thresh = np.zeros((1, sample.shape[0]))
                    arr_thresh[0, f_idx] = thresh
                    orig_thresh = scaler.inverse_transform(arr_thresh)[0, f_idx]

                    arr_val = np.zeros((1, sample.shape[0]))
                    arr_val[0, f_idx] = value
                    orig_value = scaler.inverse_transform(arr_val)[0, f_idx]
                else:
                    orig_thresh = thresh
                    orig_value = value

                direction = "<=" if orig_value <= orig_thresh else ">"
                label = f"{feat_name} ({orig_value:.2f}) {direction} {orig_thresh:.2f}"

            ax.text(0, -i, label, bbox=dict(boxstyle="round", fc="w"), ha="center")
            if i < len(node_index) - 1:
                ax.plot([0, 0], [-(i), -(i+1)], 'k-')
        ax.axis('off')
        plt.title('Used Decision Tree Path for One Sample')
        plt.savefig(f"{name}{j}.png")
        plt.close()

