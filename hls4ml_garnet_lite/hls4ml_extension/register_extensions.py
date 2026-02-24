from importlib import resources

import hls4ml

from hls4ml_garnet_lite.hls4ml_extension.garnet_lite import HGarNetLayer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_parser import parse_garnet_layer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_template import GarNetLayerConfigTemplate, GarNetLayerFunctionTemplate


def register_extensions(backend: str):
    hls4ml.converters.register_keras_v2_layer_handler('GarNetLayer', parse_garnet_layer)
    hls4ml.model.layers.register_layer('GarNetLayer', HGarNetLayer)

    backend = hls4ml.backends.get_backend(backend)
    backend.register_template(GarNetLayerConfigTemplate)
    backend.register_template(GarNetLayerFunctionTemplate)

    hls_files = resources.files('hls4ml_garnet_lite.hls')
    filenames = [
        'nnet_garnet_lite_common.h',
        'nnet_garnet_lite.h',
        'nnet_garnet_lite_stream.h',
    ]

    for fname in filenames:
        with resources.as_file(hls_files / fname) as source_path:
            backend.register_source(str(source_path))
