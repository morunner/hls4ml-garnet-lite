set(HLS4ML_GRAVNET_DIR ${CMAKE_SOURCE_DIR}/hls4ml_garnet_lite/hls)
set(HLS4ML_VIVADO_AP_TYPES_DIR ${CMAKE_SOURCE_DIR}/hls4ml/hls4ml/templates/vivado/ap_types)
set(HLS4ML_VIVADO_NNET_UTILS_DIR ${CMAKE_SOURCE_DIR}/hls4ml/hls4ml/templates/vivado/nnet_utils)

add_executable(garnet_test_runner src/test_garnet_lite.cc)

target_include_directories(garnet_test_runner PRIVATE include ${HLS4ML_GRAVNET_DIR} ${HLS4ML_VIVADO_AP_TYPES_DIR} ${HLS4ML_VIVADO_NNET_UTILS_DIR})

target_link_libraries(garnet_test_runner PRIVATE gtest_main gtest gmock pthread)

include(GoogleTest)
gtest_discover_tests(garnet_test_runner)
