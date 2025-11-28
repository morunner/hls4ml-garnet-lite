import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite import HGarNetLayer

garnetlayer_config_template = """struct config{index}: nnet::garnetlayer_config {{
static const unsigned V = {V};
static const unsigned V_nbits = {V_nbits};
static const unsigned S = {S};
static const unsigned N = {N};
static const unsigned exp_table_size = {exp_table_size};
static const unsigned exp_table_indexing_shmt = {exp_table_indexing_shmt};
}};\n"""

garnetlayer_function_template = (
    'nnet::garnetlayer<{input1_t}, {input2_t}, {output_t}, {exp_table_t}, {exp_table_idx_t},  '
    '{config}>({input1}, {input2}, {output});'
)

garnetlayer_include_list = ['nnet_utils/nnet_garnet_lite.h']


class GarNetLayerConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HGarNetLayer)
        self.template = garnetlayer_config_template

    def format(self, node):
        params = self._default_config_params(node)

        V = node.get_attr('V')
        params['V'] = V  # Number of vertices (hits)
        params['V_nbits'] = int(np.ceil(np.log2(V)))
        params['S'] = node.get_attr('S')  # Number of aggregators per vertex (hit)
        params['N'] = node.get_attr('N')  # Number of encoded features per vertex

        params['exp_table_size'] = node.get_attr('exp_table_size')
        params['exp_table_indexing_shmt'] = node.get_attr('exp_table_indexing_shmt')

        return self.template.format(**params)


class GarNetLayerFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HGarNetLayer, include_header=garnetlayer_include_list)
        self.template = garnetlayer_function_template

    def format(self, node):
        assert len(node.inputs) == 2  # Encoded features and aggregator distances
        assert len(node.outputs) == 1

        params = self._default_function_params(node)

        input_encoded_features = node.get_input_variable(node.inputs[0])
        input_aggregated_distances = node.get_input_variable(node.inputs[1])
        output = node.get_output_variable()
        exp_table_t = node.get_attr('exp_table_t')
        exp_table_idx_t = node.get_attr('exp_table_idx_t')

        # Assign types
        params['input1_t'] = input_encoded_features.type.name
        params['input2_t'] = input_aggregated_distances.type.name
        params['output_t'] = output.type.name
        params['exp_table_t'] = exp_table_t.name
        params['exp_table_idx_t'] = exp_table_idx_t.name

        # Assign input and output data args
        params['input1'] = input_encoded_features.name
        params['input2'] = input_aggregated_distances.name
        params['output'] = output.name
        params['exp_table'] = 'exp_table'

        return self.template.format(**params)
