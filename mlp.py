from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = {#'learning_rate_init': [0.01, 0.05, 0.1],
              #'learning_rate': ['constant', 'adaptive'],
              #'alpha': [0.0001, 0.005, 0.01],
              #'batch_size': [8, 16, 32, 64],
              'hidden_layer_sizes': [(128, 64, 32), (150, 100, 50)]}
new_param_grid = {'mlpclassifier__' + key: param_grid[key] for key in param_grid}


def fit(model, train_data, train_labels):
    grid_model = GridSearchCV(model, param_grid=new_param_grid, cv=3, error_score='raise')

    grid_model.fit(train_data, train_labels.values.ravel())
    return grid_model


def score(y_true, y_pred):
    print("MLP Accuracy:", accuracy_score(y_true, y_pred))
    print("MLP F1:", f1_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("MLP Precision:", precision_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("MLP Recall:", recall_score(y_true, y_pred, zero_division=0.0, average='macro'))
    print("------------------")
