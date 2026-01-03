from typing import Tuple

from keras.layers import Dense, Input
from keras.models import Model
from qkeras import QActivation, QDense, quantized_bits, quantized_relu, quantized_sigmoid

from hls4ml.contrib.garnet import GarNet, GarNetStack


class OriginalGarNetStackFactory:
    """
    This class provides the network constisting of 3 stacked
    GarNet layers as described in https://arxiv.org/abs/2008.03601
    """

    def __init__(
        self,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.precision = precision

    def create_keras_model(self):
        x = Input(shape=(128, 4))
        n = Input(shape=(1,), dtype='uint16')
        inputs = [x, n]

        v = GarNetStack(
            [4, 4, 8],
            [8, 8, 16],
            [8, 8, 16],
            simplified=True,
            collapse='mean',
            input_format='xn',
            output_activation=None,
            name='gar_1',
            quantize_transforms=True,
            total_bits=self.precision[0],
            int_bits=self.precision[1],
        )([x, n])
        v = Dense(16, activation='relu')(v)
        v = Dense(8, activation='relu')(v)
        energies = Dense(1, name='regression')(v)
        classes = Dense(1, activation='sigmoid', name='classification')(v)

        return Model(inputs=inputs, outputs=[energies, classes])


class OriginalGarNetStackFactoryTernary:
    """
    This class provides the network constisting of 3 stacked
    GarNet layers as described in https://arxiv.org/abs/2008.03601.
    GarNetStack is modified to accept ternary quantizers.
    """

    def __init__(
        self,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.precision = precision

    def create_keras_model(self):
        x = Input(shape=(128, 4))
        n = Input(shape=(1,), dtype='uint16')
        inputs = [x, n]

        v = GarNetStack(
            [4, 4, 8],
            [8, 8, 16],
            [8, 8, 16],
            simplified=True,
            collapse='mean',
            input_format='xn',
            output_activation=None,
            name='gar_1',
            quantize_transforms=True,
            total_bits=self.precision[0],
            int_bits=self.precision[1],
        )([x, n])
        v = Dense(16, activation='relu')(v)
        v = Dense(8, activation='relu')(v)
        energies = Dense(1, name='regression')(v)
        classes = Dense(1, activation='sigmoid', name='classification')(v)

        return Model(inputs=inputs, outputs=[energies, classes])


class OriginalGarNetStackFactoryFullyQuantized:
    """
    This class provides the network constisting of 3 stacked
    GarNet layers as described in https://arxiv.org/abs/2008.03601
    As opposed to the network in `OriginalGarNetStackFactory`,
    in this one all layers are fully quantized to a specified
    precision.
    """

    def __init__(
        self,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.precision = precision

    def create_keras_model(self):
        quantizer = quantized_bits(*self.precision, alpha=1)

        x = Input(shape=(128, 4))
        n = Input(shape=(1,), dtype='uint16')
        inputs = [x, n]

        v = GarNetStack(
            [4, 4, 8],
            [8, 8, 16],
            [8, 8, 16],
            simplified=True,
            collapse='mean',
            input_format='xn',
            output_activation=None,
            name='gar_1',
            quantize_transforms=True,
            total_bits=self.precision[0],
            int_bits=self.precision[1],
        )([x, n])
        v = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        v = QActivation(quantized_relu(*self.precision))(v)
        v = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        v = QActivation(quantized_relu(*self.precision))(v)
        energies = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='regression')(v)
        classes = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        classes = QActivation(quantized_sigmoid(*self.precision), name='classification')(classes)

        return Model(inputs=inputs, outputs=[energies, classes])


class OriginalGarNetFactoryFullyQuantized:
    """
    This class provides a model containing one GarNet layer
    which was implemented in https://arxiv.org/abs/2008.03601.
    As opposed to the network in `OriginalGarNetStackFactory`,
    in this one all layers are fully quantized to a specified
    precision.
    """

    def __init__(
        self,
        encoder_units: int = 16,
        aggregator_units: int = 8,
        decoder_units: int = 16,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.encoder_units = encoder_units
        self.aggregator_units = aggregator_units
        self.decoder_units = decoder_units
        self.precision = precision

    def create_keras_model(self):
        quantizer = quantized_bits(*self.precision, alpha=1)

        x = Input(shape=(128, 4))
        n = Input(shape=(1,), dtype='uint16')
        inputs = [x, n]

        v = GarNet(
            self.aggregator_units,
            self.encoder_units,
            self.decoder_units,
            simplified=True,
            collapse='mean',
            input_format='xn',
            output_activation=None,
            name='gar_1',
            quantize_transforms=True,
            int_bits=self.precision[1],
            total_bits=self.precision[0],
        )([x, n])
        v = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        v = QActivation(quantized_relu(*self.precision))(v)
        v = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        v = QActivation(quantized_relu(*self.precision))(v)
        energies = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='regression')(v)
        classes = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer)(v)
        classes = QActivation(quantized_sigmoid(*self.precision), name='classification')(classes)

        return Model(inputs=inputs, outputs=[energies, classes])
