from abc import abstractmethod
from math import log2
from typing import Tuple

import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Activation, Dense, GlobalAveragePooling1D, Layer
from qkeras import (
    QActivation,
    QDense,
    quantized_bits,
    quantized_relu,
    quantized_sigmoid,
)


class GarNetFactoryBase:
    @abstractmethod
    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
    ):
        raise NotImplementedError

    def create_keras_model(self):
        hits = Input(shape=(128, 4))

        encoded_features = self.dense_encoder(hits)
        aggregated_distances = self.dense_aggregator(hits)
        x = self.garnet([encoded_features, aggregated_distances])
        x = self.dense_decoder(x)
        x = self.activation_decoder(x)
        x = self.dense(x)
        x = self.activation(x)
        energies = self.dense_regression(x)
        classes = self.dense_classification(x)
        classes = self.activation_classification(classes)

        return Model(inputs=hits, outputs=[energies, classes])


class GarNetFactory(GarNetFactoryBase):
    def __init__(
        self,
        encoder_units: int = 16,
        aggregator_units: int = 8,
        decoder_units: int = 16,
    ):
        self.init_model(
            encoder_units,
            aggregator_units,
            decoder_units,
        )

    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
    ):
        self.dense_encoder = Dense(encoder_units)
        self.dense_aggregator = Dense(aggregator_units)
        self.garnet = GarNetLayer(name='garnet')
        self.dense_decoder = Dense(decoder_units)
        self.activation_decoder = Activation('relu')
        self.dense = Dense(8)
        self.activation = Activation('relu')
        self.dense_regression = Dense(1, name='regression')
        self.dense_classification = Dense(1)
        self.activation_classification = Activation('sigmoid', name='classification')


class QGarNetFactory(GarNetFactoryBase):
    def __init__(
        self,
        encoder_units: int = 8,
        aggregator_units: int = 4,
        decoder_units: int = 8,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.init_model(
            encoder_units=encoder_units,
            aggregator_units=aggregator_units,
            decoder_units=decoder_units,
            precision=precision,
        )

    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
        precision: Tuple[int, int] = (32, 16),
    ):
        # Currently QGarNet only supports alpha=1 due to scaling issues in HLS
        quantizer = quantized_bits(*precision, alpha=1)

        self.dense_encoder = QDense(encoder_units, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.dense_aggregator = QDense(aggregator_units, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.garnet = GarNetLayer(name='garnet')
        self.dense_decoder = QDense(decoder_units, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.activation_decoder = QActivation(quantized_relu(*precision))
        self.dense = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.activation = QActivation(quantized_relu(*precision))
        self.dense_regression = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='regression')
        self.dense_classification = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.activation_classification = QActivation(quantized_sigmoid(*precision), name='classification')


class QGarNetFactoryStacked(GarNetFactoryBase):
    def __init__(
        self,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.init_model(precision=precision)

    def init_model(
        self,
        precision: Tuple[int, int] = (32, 16),
    ):
        # Currently QGarNet only supports alpha=1 due to scaling issues in HLS
        quantizer = quantized_bits(*precision, alpha=1)

        self.encoder_1 = QDense(4, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='encoder_1')
        self.aggregator_1 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='aggregator_1')
        self.garnet_1 = GarNetLayer(name='garnet_1', collapse_mean=False)
        self.decoder_1 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='decoder_1')

        self.encoder_2 = QDense(4, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='encoder_2')
        self.aggregator_2 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='aggregator_2')
        self.garnet_2 = GarNetLayer(name='garnet_2', collapse_mean=False)
        self.decoder_2 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='decoder_2')

        self.encoder_3 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='encoder_3')
        self.aggregator_3 = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='aggregator_3')
        self.garnet_3 = GarNetLayer(name='garnet_3', collapse_mean=False)
        self.decoder_3 = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='decoder_3')

        self.avg_pool = GlobalAveragePooling1D(name='collapse_mean_pool')

        self.dense_16 = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='dense_16')
        self.act_16 = QActivation(quantized_relu(*precision), name='act_16')

        self.dense_8 = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='dense_8')
        self.act_8 = QActivation(quantized_relu(*precision), name='act_8')

        self.dense_regression = QDense(1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='regression')
        self.dense_classification = QDense(
            1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='classification_dense'
        )
        self.activation_classification = QActivation(quantized_sigmoid(*precision), name='classification')

    def create_keras_model(self, vmax=128):
        hits = Input(shape=(vmax, 4))

        encoded_1 = self.encoder_1(hits)
        agg_1 = self.aggregator_1(hits)
        g_1 = self.garnet_1([encoded_1, agg_1])
        d_1 = self.decoder_1(g_1)

        encoded_2 = self.encoder_2(d_1)
        agg_2 = self.aggregator_2(d_1)
        g_2 = self.garnet_2([encoded_2, agg_2])
        d_2 = self.decoder_2(g_2)

        encoded_3 = self.encoder_3(d_2)
        agg_3 = self.aggregator_3(d_2)
        g_3 = self.garnet_3([encoded_3, agg_3])
        d_3 = self.decoder_3(g_3)

        pooled = self.avg_pool(d_3)

        v = self.dense_16(pooled)
        v = self.act_16(v)

        v = self.dense_8(v)
        v = self.act_8(v)

        energies = self.dense_regression(v)

        classes = self.dense_classification(v)
        classes = self.activation_classification(classes)

        return Model(inputs=hits, outputs=[energies, classes])


class GarNetLayer(Layer):
    def __init__(self, V: int = 128, S: int = 4, N: int = 8, collapse_mean: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.V: int = V  # Number of vertices (hits)
        self.S: int = S  # Number of aggregators per vertex (hit)
        self.N: int = N  # Number of encoded features per vertex (hit) (coming from the encoder layer)
        self.collapse_mean = collapse_mean

    def build(self, input_shape):
        super().build(input_shape)

        shape_encoded_features, shape_aggregated_distances = input_shape

        # Ensure that both inputs contain the same number of vertices V
        assert shape_encoded_features[1] == shape_aggregated_distances[1]

        self.V = shape_aggregated_distances[1]
        self.S = shape_aggregated_distances[2]
        self.N = shape_encoded_features[2]

        if self.V > 128:
            raise ValueError('GarNetLayer currently only supports <= 128 vertices')
        if not log2(self.V).is_integer():
            raise ValueError('Number of vertices must be a power of 2')

    def call(self, inputs):
        # Unpack inputs: encoded features and aggregated distances
        fi_v, d_av = inputs

        # Weighted distances
        w_av = K.exp(-K.square(d_av))  # (B, V, S)

        # Aggregation across vertices
        fi_v = K.expand_dims(fi_v, axis=1)
        w_av_T = K.expand_dims(K.permute_dimensions(w_av, (0, 2, 1)), axis=3)
        hi_av = w_av_T * fi_v
        hi_av = K.mean(hi_av, axis=2)

        # Send aggregated features back to vertices using the same weights
        hi_av = K.expand_dims(hi_av, axis=1)
        w_av = K.expand_dims(w_av, axis=3)
        f_av_tilde = w_av * hi_av
        f_av_tilde = K.reshape(f_av_tilde, (-1, self.V, self.S * self.N))

        if self.collapse_mean:
            return tf.reduce_mean(f_av_tilde, axis=1)

        return f_av_tilde

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'V': self.V,
                'S': self.S,
                'N': self.N,
                'collapse_mean': self.collapse_mean,
            }
        )
        return config
