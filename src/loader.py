import numpy as np
import pandas as pd

from src.config import Data
from keras.utils.np_utils import to_categorical


class Loader:
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = Loader.__load_data()
        print('Data Loaded Successfully!')

    @staticmethod
    def __load_data():
        def load_file(path):
            data = pd.read_csv(filepath_or_buffer=path, header=None, delim_whitespace=True)
            return data.values

        def generate_data_path(base_address, group):
            filenames = list()
            filepath = base_address + group + '/Inertial Signals/'
            for name in Data.FEATURE_TYPE:
                for dimention in Data.DIMENSION:
                    filenames += [filepath + name + dimention + group + '.txt']

            return filenames

        def combine_features(base_address, group):
            filenames = generate_data_path(base_address, group)
            combine_data = list()
            for name in filenames:
                data = load_file(name)
                combine_data.append(data)
            combine_data = np.dstack(combine_data)
            return combine_data

        def load_dataset(base_address=Data.BASE_ADDRESS):
            x_train = combine_features(base_address, 'train')
            y_train = load_file(base_address + 'train/' + 'y_train.txt')

            x_test = combine_features(base_address, 'test')
            y_test = load_file(base_address + 'test/' + 'y_test.txt')

            y_test = y_test - 1
            y_train = y_train - 1

            y_test = to_categorical(y_test)
            y_train = to_categorical(y_train)

            return x_train, y_train, x_test, y_test

        return load_dataset()


if __name__ == '__main__':
    l = Loader()
    print(l.x_train.shape[0])
