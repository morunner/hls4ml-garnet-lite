def set_garnet_lite_hls_config(hls_config: dict) -> dict:
    hls_config['Model'] = {}
    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['Model']['Precision'] = 'ap_fixed<16,8,AP_RND,AP_SAT>'

    hls_config['LayerName']['garnet']['Precision']['result'] = 'ap_fixed<16,6,AP_RND,AP_SAT>'
    hls_config['LayerName']['garnet']['Precision']['exp_table'] = 'ap_ufixed<16,0>'
    hls_config['LayerName']['garnet']['ExponentialTable']['ScaleFactor'] = 2
    hls_config['LayerName']['garnet']['ExponentialTable']['Resolution'] = 16

    for layer in hls_config['LayerName'].keys():
        if layer != 'garnet':
            layer_precision_config = hls_config['LayerName'][layer]['Precision']
            for key in layer_precision_config.keys():
                if 'result' in key:
                    hls_config['LayerName'][layer]['Precision'][key] = 'ap_fixed<16,8>'

    hls_config['LayerName']['q_dense']['ReuseFactor'] = 32
    hls_config['LayerName']['q_dense_1']['ReuseFactor'] = 32

    return hls_config


def set_converter_opts(converter_opts: dict, backend: str = 'Vitis'):
    if backend == 'CoyoteAccelerator':
        converter_opts['io_type'] = 'io_parallel'
        converter_opts['clock_period'] = 4


def get_build_opts(backend: str = 'Vitis') -> dict:
    build_opts = {
        'csim': True,
        'synth': True,
        'cosim': True,
        'validation': True,
        'vsynth': True,
    }
    if backend == 'CoyoteAccelerator':
        build_opts['timing_opt'] = True
        build_opts['bitfile'] = True
    else:
        build_opts['vsynth'] = True

    return build_opts
