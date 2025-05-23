import os

from keras.src.regularizers import regularizers

# turns off warnings about onednn custom operations being on and available CPU instructions for potential better perf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from keras.src.callbacks import Callback
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.src.optimizers.schedules import ExponentialDecay
from keras import backend
# this is to suppress warnings about retracing tf.function calls when predicting data, this probably happens because for
# each fold in the cv a new model is created as a python object and thus tensorflow sees each model as a separate
# entity, so it cannot use cached data and has to retrace the tf.function call for the new model, note that the
# tf.function call that is retraced each fold is the predict function (one_step_on_data_distributed),
# which is wrapped in a tf.function in trainer.py:285 (found in keras package)
tf.get_logger().setLevel('ERROR')


schedules = [ExponentialDecay(initial_learning_rate=0.001, decay_steps=50, decay_rate=0.9),
             ExponentialDecay(initial_learning_rate=0.001, decay_steps=100, decay_rate=0.8),
             ExponentialDecay(initial_learning_rate=0.001, decay_steps=200, decay_rate=0.8),
             ]

param_grid = {
    'kerasclassifier__optimizer__learning_rate': schedules,
    'kerasclassifier__batch_size': [32, 64],
    'kerasclassifier__units': [32, 64],
    'kerasclassifier__activation': ['relu', 'leaky_relu'],
    'kerasclassifier__num_layers': [2, 3],
    'kerasclassifier__epochs': [100, 150],
    'kerasclassifier__l2': [0.0, 1e-5, 1e-4]
}


# fit method for hyperparameter tuning through grid search
def fit(model, train_data, train_labels):
    grid_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro',
                              error_score='raise')

    grid_model.fit(train_data, train_labels.values.ravel())
    return grid_model


# function that gets called by scikeras to build model
def create_model(meta, dropout_rate, activation,
                 units,
                 num_layers,
                 use_batchnorm1=True, use_dropout1=True,
                 use_batchnorm2=False, use_dropout2=True,
                 use_batchnorm3=True, use_dropout3=True,
                 l2=0.0):
    # meta is a dict with attributes of kerasclassifier after it is initialized, containing info like input shape,
    # number of classes etc, it is created after fit is called on the kerasclassifier and before the actual fitting
    n_features_in_ = meta["n_features_in_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))

    model.add(Dense(units, kernel_regularizer=regularizers.L2(l2=l2)))
    if use_batchnorm1:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation(activation))
    if use_dropout1:
        model.add(Dropout(dropout_rate))

    model.add(Dense(units, kernel_regularizer=regularizers.L2(l2=l2)))
    if use_batchnorm2:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation(activation))
    if use_dropout2:
        model.add(Dropout(dropout_rate))

    if num_layers == 3:
        model.add(Dense(units, kernel_regularizer=regularizers.L2(l2=l2)))
        if use_batchnorm3:
            model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        if use_dropout3:
            model.add(Dropout(dropout_rate))

    model.add(Dense(4, activation='softmax'))
    return model


# get an overview over models during cv-folds, for debugging, can be passed as callback function to KerasClassifier
class PrintModelDetails(Callback):
    def on_train_begin(self, logs=None):
        print("Model Summary:")
        self.model.summary()


def get_keras_model():
    keras_estimator = KerasClassifier(
        units=64,
        use_batchnorm1=True, use_dropout1=True,
        use_batchnorm2=False, use_dropout2=True,
        use_batchnorm3=True, use_dropout3=True,
        model=create_model,
        epochs=100,
        batch_size=32,
        verbose=0,
        activation='relu',
        dropout_rate=0.1,
        optimizer=Adam,
        num_layers=2,
        l2=0.0,
        random_state=42,
        # fit__callbacks=[PrintModelDetails(),],
        loss='sparse_categorical_crossentropy'
    )
    return keras_estimator
