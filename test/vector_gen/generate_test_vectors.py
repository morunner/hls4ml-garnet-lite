import pytest
import tensorflow as tf

from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer
from test.vector_gen.garnet_config import garnet_config
from test.vector_gen.vector import GarnetLayerTestVector
from test.vector_gen.vector_gen_config import gen_config
from test.vector_gen.vector_generator_cpp import VectorGeneratorCpp


@pytest.fixture(scope='session')
def recorder():
    rec = VectorGeneratorCpp()
    yield rec
    rec.save_to_cpp(gen_config.vector_file_path)


def test_gen_vector_garnet_layer(recorder):
    encoded_features = tf.random.normal(shape=(gen_config.n_test_vectors, garnet_config.V, garnet_config.N))
    aggregated_distances = tf.random.normal(shape=(gen_config.n_test_vectors, garnet_config.V, garnet_config.S))
    garnet_layer = GarNetLayer()

    # Trigger Keras .build()
    input_shapes = [
        (None, garnet_config.V, garnet_config.N),  # Shape of encoded_features
        (None, garnet_config.V, garnet_config.S),  # Shape of aggregated_distances
    ]
    garnet_layer.build(input_shapes)

    result = garnet_layer.call([encoded_features, aggregated_distances])

    encoded_features_np = encoded_features.numpy()
    aggregated_distances_np = aggregated_distances.numpy()
    result_np = result.numpy()

    for i in range(gen_config.n_test_vectors):
        recorder.add(
            GarnetLayerTestVector(
                name=f'garnetlayer_{i}',
                encoded_features=encoded_features_np[i],
                aggregated_distances=aggregated_distances_np[i],
                expected_result=result_np[i],
            )
        )
