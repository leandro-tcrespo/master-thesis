import os
# turns off warnings about onednn custom operations being on and available CPU instructions for potential better perf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from keras.src.callbacks import Callback
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
# this is to suppress warnings about retracing tf.function calls when predicting data, this probably happens because for
# each fold in the cv a new model is created as a python object and thus tensorflow sees each model as a separate
# entity, so it cannot use cached data and has to retrace the tf.function call for the new model, note that the
# tf.function call that is retraced each fold is the predict function (one_step_on_data_distributed),
# which is wrapped in a tf.function in trainer.py:285 (found in keras package)
tf.get_logger().setLevel('ERROR')

param_grid = {
    'kerasclassifier__num_layers': [1, 2, 3],
    'kerasclassifier__units': [32, 64, 128],
    'kerasclassifier__dropout_rate': [0.2, 0.3, 0.5],
    'kerasclassifier__optimizer__learning_rate': [0.001, 0.005],
    'kerasclassifier__batch_size': [16, 32, 64],
    'kerasclassifier__epochs': [50]
}


def fit(model, train_data, train_labels):
    grid_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro',
                              error_score='raise')

    grid_model.fit(train_data, train_labels.values.ravel())
    return grid_model


# function that gets called by scikeras to build model
def create_model(meta, units, dropout_rate, activation, num_layers):
    # meta is a dict with attributes of kerasclassifier after it is initialized, containing info like input shape,
    # number of classes etc, it is created after fit is called on the kerasclassifier and before the actual fitting
    n_features_in_ = meta["n_features_in_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))

    for _ in range(num_layers ):
        model.add(Dense(units))
        units = units // 2
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
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
        model=create_model,
        epochs=100,
        batch_size=32,
        verbose=0,
        random_state=42,
        activation='relu',
        dropout_rate=0.2,
        num_layers=1,
        units=64,
        optimizer=Adam,
        # fit__callbacks=[PrintModelDetails(),],
        loss='sparse_categorical_crossentropy',
    )
    return keras_estimator
