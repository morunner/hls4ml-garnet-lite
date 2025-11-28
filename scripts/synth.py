import argparse
import os

from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

import hls4ml
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite import HGarNetLayer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_parser import parse_garnet_layer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_template import GarNetLayerConfigTemplate, GarNetLayerFunctionTemplate
from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer
from utils.files import hls4ml_out_path, model_path, project_root
from utils.hls_config import set_garnet_lite_hls_config
from utils.training import regression_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='SynthGarNetLite', description='Synthesize GarNet Lite model with hls4ml')
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-n', '--project_name')
    parser.add_argument('-m', '--model_filename')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']

    co = {}
    _add_supported_quantized_objects(co)
    co['GarNetLayer'] = GarNetLayer
    co['regression_loss'] = regression_loss
    keras_model = load_model(model_path / args.model_filename, custom_objects=co)
    keras_model.summary()

    hls4ml.converters.register_keras_v2_layer_handler('GarNetLayer', parse_garnet_layer)
    hls4ml.model.layers.register_layer('GarNetLayer', HGarNetLayer)

    backend = hls4ml.backends.get_backend('Vitis')
    backend.register_template(GarNetLayerConfigTemplate)
    backend.register_template(GarNetLayerFunctionTemplate)
    backend.register_source(project_root / 'hls4ml_garnet_lite' / 'hls' / 'nnet_garnet_lite.h')

    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model, granularity='name', max_precision='ap_fixed<16,8,AP_RND,AP_SAT>', backend='Vitis'
    )
    set_garnet_lite_hls_config(hls_config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=hls_config,
        backend='Vitis',
        output_dir=str(hls4ml_out_path / args.project_name),
        project_name=args.project_name,
    )
    hls_model.compile()
    hls_model.build(
        csim=True,
        synth=True,
        cosim=True,
        validation=True,
        vsynth=True,
    )


if __name__ == '__main__':
    main()
