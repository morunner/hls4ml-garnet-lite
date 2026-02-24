#ifndef NNET_GARNET_LITE_H_
#define NNET_GARNET_LITE_H_

#include "hls_math.h"
#include "nnet_common.h"
#include "nnet_garnet_lite_common.h"

namespace nnet {

template <class input_T, class accum_T, typename CONFIG_T> accum_T garnetlayer_acc_tree(input_T data[CONFIG_T::V]) {
    int D_tree = CONFIG_T::V_nbits;
    int W_tree = CONFIG_T::V;
    int w_current = W_tree / 2;

    accum_T acc_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = acc_buf complete
#pragma HLS ARRAY_PARTITION variable = data complete

InitAccBuffer:
    for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
        acc_buf[v] = (accum_T)data[v];
    }

AccTreeDepth:
    for (int d = 0; d < D_tree; d++) {
#pragma HLS PIPELINE II = 1
    AccTreeWidth:
        for (int w = 0; w < W_tree / 2; w++) {
#pragma HLS UNROLL
            if (w < w_current) {
                acc_buf[w] = acc_buf[w * 2] + acc_buf[w * 2 + 1];
            }
        }
        w_current >>= 1;
    }
    return acc_buf[0];
}

template <class input1_T, class input2_T, class res_T, class accum_T, class exp_table_T, class exp_table_idx_T,
          typename CONFIG_T>
void garnet_main_loop(input1_T input1[CONFIG_T::V * CONFIG_T::N], input2_T input2[CONFIG_T::V * CONFIG_T::S],
                      exp_table_T exp_table[CONFIG_T::exp_table_size], res_T res[CONFIG_T::S * CONFIG_T::N]) {
#pragma HLS INLINE
    constexpr unsigned int REUSE = CONFIG_T::reuse;
    constexpr unsigned int BLOCK_SIZE = CONFIG_T::N / REUSE;

    res_T weight_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = weight_buf complete

    accum_T weighted_features_cache;

Aggregators:
    for (int i = 0; i < CONFIG_T::S * REUSE; i++) {
#pragma HLS PIPELINE II = 1 rewind
        int s = i / REUSE;
        int r = i % REUSE;

        if (r == 0) {
        InitWeights:
            for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
                exp_table_idx_T idx =
                    garnet_idx_from_real_val<input2_T, exp_table_idx_T, CONFIG_T>(input2[v * CONFIG_T::S + s]);
                res_T w = exp_table[idx];
                weight_buf[v] = w;
            }
            weighted_features_cache = garnetlayer_acc_tree<res_T, accum_T, CONFIG_T>(weight_buf);
        }

    Features:
        for (int n_local = 0; n_local < BLOCK_SIZE; n_local++) {
#pragma HLS UNROLL

            int n = r * BLOCK_SIZE + n_local;

            accum_T feature_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = feature_buf complete

            for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
                feature_buf[v] = (accum_T)(input1[v * CONFIG_T::N + n] * weight_buf[v]);
            }

            accum_T h = garnetlayer_acc_tree<accum_T, accum_T, CONFIG_T>(feature_buf);

            res[s * CONFIG_T::N + n] = (res_T)(h * weighted_features_cache);
        }
    }
}

template <class input1_T, class input2_T, class accum_T, class res_T, class exp_table_T, class exp_table_idx_T,
          typename CONFIG_T>
void garnetlayer(input1_T input1[CONFIG_T::V * CONFIG_T::N], input2_T input2[CONFIG_T::V * CONFIG_T::S],
                 res_T res[CONFIG_T::S * CONFIG_T::N]) {
#pragma HLS ARRAY_PARTITION variable = input1 type = cyclic factor = CONFIG_T::N
#pragma HLS ARRAY_PARTITION variable = input2 type = cyclic factor = CONFIG_T::S
#pragma HLS ARRAY_PARTITION variable = res type = block factor = CONFIG_T::S

#ifdef __HLS_SYN__
    bool initialized = false;
    exp_table_T exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#endif

    if (!initialized) {
        garnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);
        initialized = true;
    }

    garnet_main_loop<input1_T, input2_T, res_T, accum_T, exp_table_T, exp_table_idx_T, CONFIG_T>(input1, input2, exp_table,
                                                                                                 res);
}

} // namespace nnet

#endif // NNET_GARNET_LITE_H_
