from dataclasses import dataclass


@dataclass(frozen=True)
class GenerateTestVectorsConfig:
    vector_file_path: str = 'test/hls/include/test_vectors.h'
    n_test_vectors: int = 4


gen_config = GenerateTestVectorsConfig()
