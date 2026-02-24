from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer


def parse_garnet_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'GarNetLayer'
    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['V'] = keras_layer['config']['V']  # Number of vertices (hits)
    layer['S'] = keras_layer['config']['S']  # Number of aggregators per vertex (hit)
    layer['N'] = keras_layer['config']['N']  # Number of encoded features per vertex

    collapse_mean = keras_layer['config']['collapse_mean']
    layer['collapse_mean'] = collapse_mean
    if collapse_mean:
        output_shape = [None, layer['S'] * layer['N']]
    else:
        output_shape = [None, layer['V'], layer['S'] * layer['N']]
    return layer, output_shape
