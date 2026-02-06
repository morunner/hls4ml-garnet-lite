# hls4ml-garnet-lite

This is a smaller, unofficial version of the GarNet architecture originally proposed by Iiyama et al.

The original implementation is located at [yiiyama/graph-hls-paper](https://github.com/yiiyama/graph-hls-paper). Original source code and configurations can also be found in the [./legacy](./legacy/) directory of this repository.

### Key Changes
* Only supports one GarNet layer (no `GarNetStack`).
* Only includes core graph operations in `GarNetLayer`.
* Reuses standard Keras `Dense` layers for the Encoder, Aggregator Distances, and Decoder.

## Getting Started

### Prerequisites

[uv](https://docs.astral.sh/uv/) is used as the package manager. Make sure it is available on your system.

### Installation

To create a virtual environment and install the `hls4ml_garnet_lite` package, run:

```bash
uv sync

```

If you want to install it as an editable package:

```bash
uv pip install -e .

```

If you already have a project and want to add this package to your existing virtual environment:

```bash
uv add git+[https://github.com/morunner/hls4ml-garnet-lite.git](https://github.com/morunner/hls4ml-garnet-lite.git)

```

## Usage

### Keras

The custom layer `GarNetLayer` (which contains no trainable weights) can be imported and used for training graph neural networks.

Example models are available in the [keras_model](https://www.google.com/search?q=./hls4ml_garnet_lite/keras_model) directory.

* `garnet_original.py`: Factories for the original implementation.
* `garnet_lite.py`: Contains our "lite" version of the model.

Fully quantized models using [QKeras](https://github.com/google/qkeras) are supported. A model can be created from one of the factories:

```python
factory = QGarNetFactory(16, 8, 16, precision=(8,0))
model = factory.create_keras_model()
```

To reload a trained model from a `*.keras` file, you must register the custom objects:

```python
from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer
from hls4ml.contrib.garnet import GarNet, GarNetStack
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.models import load_model

co = {}
_add_supported_quantized_objects(co)
co.update({'GarNet': GarNet, 'GarNetStack': GarNetStack, 'GarNetLayer': GarNetLayer})

model = load_model("path/to/model.keras", custom_objects=co)
```

### hls4ml

This project is intended to be used with [hls4ml](https://github.com/fastmachinelearning/hls4ml). To convert a model from Keras to HLS, we use the hls4ml [Extension API](https://fastmachinelearning.org/hls4ml/advanced/extension.html).

Here is an example of converting the `QGarNetFactory` model to HLS using the **Vitis** backend:

```python
import hls4ml
from hls4ml_garnet_lite.hls4ml_extension.register_extensions import register_extensions
from hls4ml_garnet_lite.hls4ml_extension.hls_config import set_garnet_lite_hls_config

# Register the custom layer
register_extensions(backend='Vitis')

# Configure HLS
hls_config = hls4ml.utils.config_from_keras_model(model)
set_garnet_lite_hls_config(hls_config=hls_config, garnet_reuse=1) # ReuseFactor of 1

# Convert to HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    model=model,
    hls_config=hls_config,
    backend='Vitis',
    output_dir='my_hls_project'
)
hls_model.compile()

# Synthesize
hls_model.build()
```

## Reference

This project is based on the architecture described in the following paper:

> **Distance-Weighted Graph Neural Networks on FPGAs for Real-Time Particle Reconstruction in High Energy Physics**
> Yutaro Iiyama, et al.
> *Frontiers in Big Data, 2021.*
> [[DOI]](http://dx.doi.org/10.3389/fdata.2020.598927)

If you use this work, please cite the original paper:

```bibtex
@article{Iiyama_2021,
   title={Distance-Weighted Graph Neural Networks on FPGAs for Real-Time Particle Reconstruction in High Energy Physics},
   volume={3},
   ISSN={2624-909X},
   url={[http://dx.doi.org/10.3389/fdata.2020.598927](http://dx.doi.org/10.3389/fdata.2020.598927)},
   DOI={10.3389/fdata.2020.598927},
   journal={Frontiers in Big Data},
   publisher={Frontiers Media SA},
   author={Iiyama, Yutaro and Cerminara, Gianluca and Gupta, Abhijay and Kieseler, Jan and Loncar, Vladimir and Pierini, Maurizio and Qasim, Shah Rukh and Rieger, Marcel and Summers, Sioni and Van Onsem, Gerrit and Wozniak, Kinga Anna and Ngadiuba, Jennifer and Di Guglielmo, Giuseppe and Duarte, Javier and Harris, Philip and Rankin, Dylan and Jindariani, Sergo and Liu, Mia and Pedro, Kevin and Tran, Nhan and Kreinar, Edward and Wu, Zhenbin},
   year={2021},
   month=jan
}
