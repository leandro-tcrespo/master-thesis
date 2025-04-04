import os
# turns off warnings about onednn custom operations being on and available CPU instructions for potential better perf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam, SGD, RMSprop
from keras.src.callbacks import Callback, EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.src.optimizers.schedules import ExponentialDecay
# this is to suppress warnings about retracing tf.function calls when predicting data, this probably happens because for
# each fold in the cv a new model is created as a python object and thus tensorflow sees each model as a separate
# entity, so it cannot use cached data and has to retrace the tf.function call for the new model, note that the
# tf.function call that is retraced each fold is the predict function (one_step_on_data_distributed),
# which is wrapped in a tf.function in trainer.py:285 (found in keras package)
tf.get_logger().setLevel('ERROR')


lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=50, decay_rate=0.9)

adam_with_decay = Adam(learning_rate=lr_schedule)
sgd_with_decay = SGD(learning_rate=lr_schedule, momentum=0.9)
rmsprop_with_decay = RMSprop(learning_rate=lr_schedule)

early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True
)

param_grid = {
    'kerasclassifier__optimizer': [adam_with_decay, sgd_with_decay, rmsprop_with_decay],
    'kerasclassifier__dropout_rate': [0.1, 0.15, 0.2],
    'kerasclassifier__batch_size': [32, 64],
    'kerasclassifier__use_batchnorm1': [True, False],
    'kerasclassifier__use_dropout1': [True, False],
    'kerasclassifier__use_batchnorm2': [True, False],
    'kerasclassifier__use_dropout2': [True, False],
    'kerasclassifier__use_batchnorm3': [True, False],
    'kerasclassifier__use_dropout3': [True, False],
    'kerasclassifier__activation': ['relu', 'leaky_relu', 'elu', 'silu'],
}


def fit(model, train_data, train_labels):
    grid_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro',
                              error_score='raise')

    grid_model.fit(train_data, train_labels.values.ravel())
    return grid_model


# function that gets called by scikeras to build model
def create_model(meta, dropout_rate, activation,
                 units1, units2, units3,
                 use_batchnorm1, use_dropout1,
                 use_batchnorm2, use_dropout2,
                 use_batchnorm3, use_dropout3):
    # meta is a dict with attributes of kerasclassifier after it is initialized, containing info like input shape,
    # number of classes etc, it is created after fit is called on the kerasclassifier and before the actual fitting
    n_features_in_ = meta["n_features_in_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))

    model.add(Dense(units1))
    if use_batchnorm1:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation(activation))
    if use_dropout1:
        model.add(Dropout(dropout_rate))

    model.add(Dense(units2))
    if use_batchnorm2:
        model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation(activation))
    if use_dropout2:
        model.add(Dropout(dropout_rate))

    model.add(Dense(units3))
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
        units1=64,
        units2=64,
        units3=64,
        use_batchnorm1=False, use_dropout1=False,
        use_batchnorm2=False, use_dropout2=False,
        use_batchnorm3=False, use_dropout3=False,
        model=create_model,
        epochs=200,
        batch_size=32,
        verbose=0,
        random_state=42,
        activation='relu',
        dropout_rate=0.2,
        optimizer=Adam,
        callbacks=[early_stopping],
        # fit__callbacks=[PrintModelDetails(),],
        loss='sparse_categorical_crossentropy',
    )
    return keras_estimator
