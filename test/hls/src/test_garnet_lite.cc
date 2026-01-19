#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_garnet_lite.h"
#include "test_vectors.h"
#include "gtest/gtest.h"
#include <cstddef>

TEST(Hls4mlGarNetTest, garnet_layer) {
    const double abs_error = 0.05;
    typedef ap_fixed<16, 8> encoded_features_t;
    typedef ap_fixed<16, 8> aggregated_distances_t;
    typedef ap_ufixed<16, 1> exp_table_t;
    typedef ap_uint<garnet_config::exp_table_size_nbits> exp_table_idx_t;
    typedef ap_fixed<16, 6> result_t;

    for (size_t i = 0; i < garnet_layer_test_vectors_length; i++) {
        garnet_layer_test_vector v = garnet_layer_test_vectors[i];

        encoded_features_t encoded_features[v.encoded_features_len];
        for (unsigned int i = 0; i < v.encoded_features_len; i++) {
            encoded_features[i] = v.encoded_features[i];
        }
        aggregated_distances_t aggregated_distances[v.aggregated_distances_len];
        for (unsigned int i = 0; i < v.aggregated_distances_len; i++) {
            aggregated_distances[i] = v.aggregated_distances[i];
        }
        result_t actual_result[garnet_config::S * garnet_config::N];
        nnet::garnetlayer<encoded_features_t, aggregated_distances_t, result_t, exp_table_t, exp_table_idx_t, garnet_config>(
            encoded_features, aggregated_distances, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}
