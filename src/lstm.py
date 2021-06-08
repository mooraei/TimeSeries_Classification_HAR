import os

import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from src.config import Model
from src.loader import Loader

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run():
    data = Loader()

    n_time_steps = data.x_train.shape[1]
    n_output = data.y_train.shape[1]
    n_features = data.x_train.shape[2]

    model = LSTMModel(n_time_steps, n_output, n_features)
    model.fit(x_train=data.x_train, y_train=data.y_train)

    accuracy = model.evaluate(x_test=data.x_test, y_test=data.y_test)
    print('Accuracy = %{0:.4f}'.format(accuracy*100))
    print(model.model.summary())


class LSTMModel:
    def __init__(self, n_time_steps, n_output, n_features):
        self.model = Sequential()
        self.model.add(LSTM(units=Model.N_NODES, input_shape=(n_time_steps, n_features)))
        # Use Dropout to reduce overfitting
        self.model.add(Dropout(Model.DROPOUT_RATE))

        self.model.add(Dense(Model.N_NODES, activation='relu'))
        self.model.add(Dense(n_output, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, epoch=15, batch_size=64, verbose=0):
        self.model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, verbose=verbose)

    def predict(self, x_test, batch_size=64):
        return self.model.predict(x=x_test, batch_size=batch_size)

    def evaluate(self, x_test, y_test, batch_size=64, verbose=0):
        _, accuracy = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=verbose)
        return accuracy


if __name__ == '__main__':
    run()