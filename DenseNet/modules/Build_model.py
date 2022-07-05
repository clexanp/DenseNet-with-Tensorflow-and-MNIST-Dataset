from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
import tensorflow as tf

def Build_net(shape, dense_block_num, growthRate):

    def Build_conv2d(x, channel, stride):
        return (Conv2D(channel, kernel_size=(3, 3), padding='same',
                       kernel_initializer='he_uniform', use_bias=False,
                       strides=stride, activation=None))(x)

    def Build_Dense_block(name, x):
        shape = x.get_shape().as_list()
        input_channel = shape[3]
        with tf.compat.v1.variable_scope(name) as scope:
            dense = BatchNormalization()(x)
            dense = Activation('relu')
            dense = Build_conv2d(x, growthRate, (1, 1))
            return tf.concat([x, dense], 3)

    def Build_transition(name, x):
        shape = x.get_shape().as_list()
        input_channel = shape[3]
        with tf.compat.v1.variable_scope(name) as scope:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(input_channel, kernel_size=(1, 1),
                       activation=None, strides=(1, 1),
                       use_bias=False)(x)
            x = AveragePooling2D([2, 2])(x)
            return x

    channel = 32
    row, col = shape[0], shape[1]
    input_layer = Input(shape=(row, col, 1), name='input')

    x = Build_conv2d(input_layer, channel, (1, 1))

    name_1, name_2, name_3 = 'dense_block_1', 'dense_block_2', 'dense_block_3'

    """
    ====================================
    =========Build First Block==========
    ====================================
    """

    with tf.name_scope(name_1) as scope:

        for i in range(dense_block_num):
            x = Build_Dense_block('dense_layer_{num}'.format(num=i), x)

        x = Build_transition('transtion_1', x)

    """
    ====================================
    =========Build Second Block=========
    ====================================
    """

    with tf.name_scope(name_2) as scope:

        for i in range(dense_block_num):
            x = Build_Dense_block('dense_layer_{num}'.format(num=i), x)

        x = Build_transition('transtion_2', x)

    """
    ====================================
    =========Build Third Block==========
    ====================================
    """

    with tf.name_scope(name_3) as scope:

        for i in range(dense_block_num):
            x = Build_Dense_block('dense_layer_{num}'.format(num=i), x)

        x = Build_transition('transtion_3', x)

    x = BatchNormalization(name='bn_last')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    out = Dense(10, activation='softmax')(x)
    model = Model(input_layer, out)

    return model
