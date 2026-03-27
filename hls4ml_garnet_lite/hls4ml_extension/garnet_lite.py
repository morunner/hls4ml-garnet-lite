from math import ceil, log2

from hls4ml.model.attributes import Attribute, TypeAttribute
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType


class HGarNetLayer(Layer):
    _expected_attributes = [
        Attribute('V'),
        Attribute('S'),
        Attribute('N'),
        Attribute('exponential_table', value_type=dict, default={'ScaleFactor': 2, 'Resolution': 16}, configurable=True),
        Attribute('exp_table_size', value_type=int),
        Attribute('exp_table_indexing_shmt', value_type=int),
        TypeAttribute('exp_table', default=FixedPrecisionType(width=16, integer=0, signed=False), configurable=True),
        TypeAttribute('exp_table_idx', default=IntegerPrecisionType(width=4, signed=False)),
    ]

    def initialize(self):
        exp_table_params = self.get_attr('exponential_table', None)
        scale_factor = exp_table_params['ScaleFactor']
        resolution = exp_table_params['Resolution']

        if not log2(resolution).is_integer():
            raise ValueError('Exponential table resolution must be a power of two')
        if not log2(scale_factor).is_integer():
            raise ValueError('Scale factor must be a power of two')

        exp_table_size = scale_factor * resolution
        exp_table_indexing_shmt = int(ceil(log2(resolution)))
        exp_table_size_nbits = int(ceil(log2(exp_table_size)))

        self.set_attr('exp_table_size', exp_table_size)
        self.set_attr('exp_table_indexing_shmt', exp_table_indexing_shmt)
        self.set_attr('exp_table_idx_t', IntegerPrecisionType(width=exp_table_size_nbits, signed=False))

        self._set_type_t('exp_table')

        shape = [self.get_attr('V')]
        self.add_output_variable(shape)
