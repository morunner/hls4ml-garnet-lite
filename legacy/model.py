import numpy as np
try:
    import tensorflow.keras as keras
except ImportError:
    import keras
K = keras.backend

from legacy.garnet import GarNetStack

def make_model(vmax, quantize):
    x = keras.layers.Input(shape=(vmax, 4))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    inputs = [x, n]

    v = GarNetStack([4, 4, 8], [8, 8, 16], [8, 8, 16], simplified=True, collapse='mean', input_format='xn', output_activation=None, name='gar_1', quantize_transforms=quantize)([x, n])
    v = keras.layers.Dense(16, activation='relu')(v)
    v = keras.layers.Dense(8, activation='relu')(v)
    v1 = keras.layers.Dense(1, name='regression')(v)
    v2 = keras.layers.Dense(1, activation='sigmoid', name='classification')(v)
    outputs = [v1, v2]
    
    return keras.Model(inputs=inputs, outputs=outputs)

def regression_loss(y_true, y_pred):
    with K.name_scope('regression_loss'):
        y_true /= 100. # because our data is max 100 GeV

        return K.mean(K.square((y_true - y_pred) / y_true), axis=-1)

classification_loss = 'binary_crossentropy'
