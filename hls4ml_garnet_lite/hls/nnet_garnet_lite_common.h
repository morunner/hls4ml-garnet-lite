#ifndef NNET_GARNET_LITE_COMMON_H_
#define NNET_GARNET_LITE_COMMON_H_

#include "hls_math.h"
#include "nnet_common.h"

namespace nnet {

struct garnetlayer_config {
    static const unsigned V = 128;
    static const unsigned V_nbits = 7;
    static const unsigned S = 8;
    static const unsigned N = 16;
    static const unsigned exp_table_size = 32;
    static const unsigned exp_table_indexing_shmt = 4;
    static const unsigned reuse = 1;
};

inline float garnet_exp_fcn_float(float input) { return std::exp(input); }

template <class data_T, class exp_table_idx_T, typename CONFIG_T> inline exp_table_idx_T garnet_idx_from_real_val(data_T x) {
    if (x < 0)
        x = -x;

    exp_table_idx_T max_idx = CONFIG_T::exp_table_size - 1;
    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);
    if (idx > max_idx) {
        return max_idx;
    }
    return (exp_table_idx_T)idx;
}

template <class exp_table_T, typename CONFIG_T> void garnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1.0f;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0.0f;

    for (unsigned i = 1; i < CONFIG_T::exp_table_size - 1; i++) {
#pragma HLS UNROLL
        float val = (float)((ap_fixed<32, 16>)(i + 1) >> CONFIG_T::exp_table_indexing_shmt);
        exp_table_T exp_x = garnet_exp_fcn_float(-val * val);
        table_out[i] = exp_x;
    }
}

} // namespace nnet
#endif
