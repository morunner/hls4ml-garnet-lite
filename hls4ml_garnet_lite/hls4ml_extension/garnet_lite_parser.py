from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer


def parse_garnet_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'GarNetLayer'
    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['V'] = keras_layer['config']['V']  # Number of vertices (hits)
    layer['S'] = keras_layer['config']['S']  # Number of aggregators per vertex (hit)
    layer['N'] = keras_layer['config']['N']  # Number of encoded features per vertex

    output_shape = [input_shapes[0][1]]
    return layer, output_shape
