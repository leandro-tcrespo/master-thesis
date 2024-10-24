import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

data = pd.read_csv("./Synthetic_data.csv", header=0)
filtered_data = data.drop(['diag_binary'], axis=1)

X = filtered_data[['sex', 'age', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']]
y = filtered_data[['diag_multi']]

datasets = train_test_split(X, y, test_size=0.25, random_state=42)
train_data, test_data, train_labels, test_labels = datasets

enc = ColumnTransformer([("OneHot", OneHotEncoder(handle_unknown='ignore',),
                        ['sex', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q'])], remainder='passthrough', sparse_threshold=0)
train_data = enc.fit_transform(train_data)
test_data = enc.transform(test_data)

smote_os = SMOTE(random_state=42, k_neighbors=1)
random_os = RandomOverSampler(random_state=42)

param_grid = {'learning_rate_init': [0.01, 0.05, 0.1],
              'learning_rate': ['constant', 'adaptive'],
              'alpha': [0.0001, 0.005, 0.01],
              'batch_size': [8, 16, 32, 64],
              'hidden_layer_sizes': [(128, 64, 32), (150, 100, 50)]}
new_param_grid = {'mlpclassifier__' + key: param_grid[key] for key in param_grid}

mlp = MLPClassifier(max_iter=1000, random_state=42)
imba_pipeline_smote = make_pipeline(smote_os, mlp)
imba_pipeline_random = make_pipeline(random_os, mlp)
grid_mlp_smote = GridSearchCV(imba_pipeline_smote, param_grid=new_param_grid, cv=3, verbose=4, error_score='raise')
grid_mlp_random = GridSearchCV(imba_pipeline_random, param_grid=new_param_grid, cv=3, verbose=4, error_score='raise')

grid_mlp_smote.fit(train_data, train_labels.values.ravel())
grid_mlp_random.fit(train_data, train_labels.values.ravel())

print("------------------")
print("SMOTE MLP results:")
predictions_smote = grid_mlp_smote.predict(test_data)
print("Train accuracy:", grid_mlp_smote.score(train_data, train_labels))
print("Test accuracy:", grid_mlp_smote.score(test_data, test_labels))
print("Test F1:", f1_score(test_labels, predictions_smote, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))
print("Test Precision:", precision_score(test_labels, predictions_smote, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))
print("Test Recall:", recall_score(test_labels, predictions_smote, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))

print("------------------")
print("RandomOS MLP results:")
predictions_random = grid_mlp_random.predict(test_data)
print("Train accuracy:", grid_mlp_random.score(train_data, train_labels))
print("Test accuracy:", grid_mlp_random.score(test_data, test_labels))
print("Test F1:", f1_score(test_labels, predictions_random, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))
print("Test Precision:", precision_score(test_labels, predictions_random, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))
print("Test Recall:", recall_score(test_labels, predictions_random, labels=grid_mlp_smote.classes_, zero_division=0.0, average='macro'))
print("------------------")
