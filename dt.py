from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['sqrt', 'log2', None],
    'min_impurity_decrease': [0.0, 0.01],
    'ccp_alpha': [0.0, 0.01],
    'max_leaf_nodes': [10, 20, 50, 100, 200, None]
}
new_param_grid = {'decisiontreeclassifier__' + key: param_grid[key] for key in param_grid}


def fit(model, t_data, t_labels):
    grid_model = GridSearchCV(model, param_grid=new_param_grid, cv=5, n_jobs=-1, scoring='f1_macro',
                              error_score='raise')

    grid_model.fit(t_data, t_labels.values.ravel())
    return grid_model


def plot_dt(grid_dt, out_file):
    plt.figure(figsize=(15, 10))
    plot_tree(
        grid_dt.best_estimator_['decisiontreeclassifier'],
        feature_names=grid_dt.best_estimator_['columntransformer'].get_feature_names_out(),
        class_names=grid_dt.best_estimator_.classes_.tolist(),
        filled=True,
        rounded=True
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")  # Save as PNG
    plt.show()


def plot_dt_without_grid(dt, out_file):
    plt.figure(figsize=(15, 10))
    plot_tree(
        dt['decisiontreeclassifier'],
        feature_names=dt['columntransformer'].get_feature_names_out(),
        class_names=dt.classes_.tolist(),
        filled=True,
        rounded=True
    )

    plt.savefig(out_file, dpi=300, bbox_inches="tight")  # Save as PNG
    # plt.show()
