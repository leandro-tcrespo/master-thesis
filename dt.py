from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['sqrt', None],
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
    plt.savefig(out_file, dpi=300, bbox_inches="tight")  # Save as PNG
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

    plt.savefig(out_file, dpi=300, bbox_inches="tight")  # Save as PNG
    # plt.show()
