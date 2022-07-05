import numpy as np

class MnistData():
    def __init__(self, path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        img_rows, img_cols = x_train.shape[1:]

        x_train = x_train.astype('float32') - 127.5
        x_test = x_test.astype('float32') - 127.5
        x_train /= 127.5
        x_test /= 127.5

        self.num_classes = 10
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
