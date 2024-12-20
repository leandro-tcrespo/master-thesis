from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

param_grid = {'criterion': ["gini", "entropy", "log_loss"],
              'max_depth': [10, 100, 1000]}
new_param_grid = {'decisiontreeclassifier__' + key: param_grid[key] for key in param_grid}


def fit(model, t_data, t_labels):
    grid_model = GridSearchCV(model, param_grid=new_param_grid, cv=3, error_score='raise')

    grid_model.fit(t_data, t_labels.values.ravel())
    return grid_model


def score(y_true, y_pred):
    print("DT Accuracy:", accuracy_score(y_true, y_pred))
    print("DT F1:", f1_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("DT Precision:", precision_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("DT Recall:", recall_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("------------------")


def plot_dt(grid_dt):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(grid_dt.best_estimator_['decisiontreeclassifier'], filled=True,
                   feature_names=grid_dt.best_estimator_['columntransformer'].get_feature_names_out(),
                   class_names=grid_dt.best_estimator_.classes_.tolist())
    plt.tight_layout()
    plt.show()
