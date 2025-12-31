import argparse
import os

from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from sklearn.metrics import roc_auc_score

import hls4ml
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite import HGarNetLayer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_parser import parse_garnet_layer
from hls4ml_garnet_lite.hls4ml_extension.garnet_lite_template import (
    GarNetLayerConfigTemplate,
    GarNetLayerFunctionTemplate,
)
from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer
from hls4ml_garnet_lite.utils.data import load_data
from hls4ml_garnet_lite.utils.evaluation import garnet_predict, response_rmse
from hls4ml_garnet_lite.utils.files import hls4ml_out_path, model_path, project_root
from hls4ml_garnet_lite.utils.hls_config import get_build_opts, set_converter_opts, set_garnet_lite_hls_config
from hls4ml_garnet_lite.utils.training import regression_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='SynthGarNetLite', description='Synthesize GarNet Lite model with hls4ml')
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-viv', '--vivado_path', default='')  # Optional
    parser.add_argument('-n', '--project_name')
    parser.add_argument('-o', '--output_dir', default='')
    parser.add_argument('-m', '--model_filename')
    parser.add_argument('-r', '--reuse', type=int)
    parser.add_argument('-b', '--backend', default='Vitis')

    return parser.parse_args()


def run_garnet_synthesis(args: argparse.Namespace):
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']
    os.environ['PATH'] = args.vivado_path + 'bin:' + os.environ['PATH']

    co = {}
    _add_supported_quantized_objects(co)
    co['GarNetLayer'] = GarNetLayer
    co['regression_loss'] = regression_loss
    keras_model = load_model(model_path / args.model_filename, custom_objects=co)
    keras_model.summary()

    hls4ml.converters.register_keras_v2_layer_handler('GarNetLayer', parse_garnet_layer)
    hls4ml.model.layers.register_layer('GarNetLayer', HGarNetLayer)

    backend = hls4ml.backends.get_backend(args.backend)
    backend.register_template(GarNetLayerConfigTemplate)
    backend.register_template(GarNetLayerFunctionTemplate)
    backend.register_source(project_root / 'hls4ml_garnet_lite' / 'hls' / 'nnet_garnet_lite.h')

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', default_reuse_factor=1)
    set_garnet_lite_hls_config(hls_config=hls_config, garnet_reuse=1)
    hls_config['LayerName']['q_dense']['ReuseFactor'] = args.reuse
    hls_config['LayerName']['q_dense_1']['ReuseFactor'] = args.reuse

    converter_opts = {
        'model': keras_model,
        'hls_config': hls_config,
        'backend': args.backend,
        'output_dir': str(hls4ml_out_path / args.output_dir / args.project_name),
        'project_name': args.project_name,
    }
    set_converter_opts(converter_opts, args.backend)
    build_opts = get_build_opts(args.backend)

    hls_model = hls4ml.converters.convert_from_keras_model(**converter_opts)
    hls_model.compile()

    # Check accuracies
    _, _, X_test, y_test = load_data(suffix='baseline_1')
    if 'original' not in args.model_filename:
        X_test = X_test[0]
    test_energy_pred_hls, test_pid_pred_hls = garnet_predict(hls_model, X_test)
    test_energy_pred_hls *= 100
    test_response_rmse = response_rmse(y_test[0], test_energy_pred_hls)
    test_auc = roc_auc_score(y_test[1], test_pid_pred_hls)
    print(f'HLS model RMSE={test_response_rmse:.2f} AUC={test_auc:.2f} for reuse factor {args.reuse}')

    hls_model.build(**build_opts)


if __name__ == '__main__':
    args = parse_args()
    run_garnet_synthesis(args)
