def set_garnet_lite_hls_config(hls_config: dict, garnet_reuse: int) -> dict:
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,8,AP_RND,AP_SAT>', 'maximum': 'ap_fixed<16,8,AP_RND,AP_SAT>'}

    hls_config['LayerName']['garnet']['Precision']['result'] = 'ap_fixed<16,6,AP_RND,AP_SAT>'
    hls_config['LayerName']['garnet']['Precision']['exp_table'] = 'ap_ufixed<16,1>'
    hls_config['LayerName']['garnet']['ExponentialTable']['ScaleFactor'] = 2
    hls_config['LayerName']['garnet']['ExponentialTable']['Resolution'] = 16
    hls_config['LayerName']['garnet']['ReuseFactor'] = garnet_reuse
