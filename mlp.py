from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'early_stopping': [True]
}
new_param_grid = {'mlpclassifier__' + key: param_grid[key] for key in param_grid}


def fit(model, train_data, train_labels):
    grid_model = GridSearchCV(model, param_grid=new_param_grid, cv=5, n_jobs=-1, error_score='raise')

    grid_model.fit(train_data, train_labels.values.ravel())
    return grid_model
