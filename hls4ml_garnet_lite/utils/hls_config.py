def set_garnet_lite_hls_config(hls_config: dict, garnet_reuse: int) -> dict:
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,8,AP_RND,AP_SAT>', 'maximum': 'ap_fixed<16,8,AP_RND,AP_SAT>'}

    hls_config['LayerName']['garnet']['Precision']['result'] = 'ap_fixed<16,6,AP_RND,AP_SAT>'
    hls_config['LayerName']['garnet']['Precision']['exp_table'] = 'ap_ufixed<16,1>'
    hls_config['LayerName']['garnet']['ExponentialTable']['ScaleFactor'] = 2
    hls_config['LayerName']['garnet']['ExponentialTable']['Resolution'] = 16
    hls_config['LayerName']['garnet']['ReuseFactor'] = garnet_reuse


def set_converter_opts(converter_opts: dict, backend: str = 'Vitis'):
    if backend == 'Vitis':
        converter_opts['part'] = 'xcku115-flvb2104-2-i'
    elif backend == 'CoyoteAccelerator':
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
