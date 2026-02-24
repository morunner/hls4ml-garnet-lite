#ifndef NNET_GARNET_STREAM_H_
#define NNET_GARNET_STREAM_H_

#include "hls_math.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_garnet_lite_common.h"

namespace nnet {

template <class input2_T, class weight_T, class exp_table_T, class exp_table_idx_T, typename CONFIG_T>
void garnet_compute_weights(hls::stream<input2_T> &input2_stream,
                            hls::stream<nnet::array<weight_T, CONFIG_T::S>> &weight_stream,
                            weight_T weight_cache[CONFIG_T::V][CONFIG_T::S],
                            const exp_table_T exp_table[CONFIG_T::exp_table_size]) {
    typedef typename input2_T::value_type in2_val_t;

ComputeWeights:
    for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS PIPELINE II = 1
        input2_T in2_word = input2_stream.read();
        nnet::array<weight_T, CONFIG_T::S> w_out;

        for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            exp_table_idx_T idx = garnet_idx_from_real_val<in2_val_t, exp_table_idx_T, CONFIG_T>(in2_word[s]);
            weight_T w = exp_table[idx];

            weight_cache[v][s] = w;
            w_out[s] = w;
        }
        weight_stream.write(w_out);
    }
}

template <class input1_T, class weight_T, class accum_T, typename CONFIG_T>
void garnet_accumulate(hls::stream<input1_T> &input1_stream, hls::stream<nnet::array<weight_T, CONFIG_T::S>> &weight_stream,
                       accum_T h[CONFIG_T::S][CONFIG_T::N]) {
InitAccumulators:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
        for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS UNROLL
            h[s][n] = 0;
        }
    }

AccumulateFeatures:
    for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS PIPELINE II = 1
        input1_T in1_word = input1_stream.read();
        nnet::array<weight_T, CONFIG_T::S> w_in = weight_stream.read();

        for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS UNROLL
                h[s][n] += (accum_T)in1_word[n] * (accum_T)w_in[s];
            }
        }
    }
}

template <class output_T, class accum_T, class weight_T, typename CONFIG_T>
void garnet_broadcast_and_output(const weight_T weight_cache[CONFIG_T::V][CONFIG_T::S],
                                 const accum_T h[CONFIG_T::S][CONFIG_T::N], hls::stream<output_T> &res_stream) {
    typedef typename output_T::value_type output_val_t;

WriteOut:
    for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS PIPELINE II = 1
        output_T out_features;

        for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            weight_T w = weight_cache[v][s];

            for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS UNROLL
                accum_T out_feature = (h[s][n] * (accum_T)w) / (accum_T)CONFIG_T::V;
                out_features[s * CONFIG_T::N + n] = (output_val_t)out_feature;
            }
        }
        res_stream.write(out_features);
    }
}

template <class input1_T, class input2_T, class accum_T, class output_T, class exp_table_T, class exp_table_idx_T,
          typename CONFIG_T>
void garnetlayer(hls::stream<input1_T> &input1_stream, hls::stream<input2_T> &input2_stream,
                 hls::stream<output_T> &res_stream) {
#pragma HLS DATAFLOW

    typedef typename output_T::value_type res_val_t;

    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    garnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);

    res_val_t weight_cache[CONFIG_T::V][CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 2

    accum_T h[CONFIG_T::S][CONFIG_T::N];
#pragma HLS ARRAY_PARTITION variable = h complete dim = 0

    hls::stream<nnet::array<res_val_t, CONFIG_T::S>> weight_stream("weight_stream");
#pragma HLS STREAM variable = weight_stream depth = 4

    garnet_compute_weights<input2_T, res_val_t, exp_table_T, exp_table_idx_T, CONFIG_T>(input2_stream, weight_stream,
                                                                                        weight_cache, exp_table);

    garnet_accumulate<input1_T, res_val_t, accum_T, CONFIG_T>(input1_stream, weight_stream, h);

    garnet_broadcast_and_output<output_T, accum_T, res_val_t, CONFIG_T>(weight_cache, h, res_stream);
}

} // namespace nnet

#endif // NNET_GARNET_STREAM_H_
