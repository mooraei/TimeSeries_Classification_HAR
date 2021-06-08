from src.loader import Loader


def runner():
    data = Loader()
    print(data.x_train[0])
    print('==============================')
    print(data.x_train[0].shape)


if __name__ == '__main__':
    runner()