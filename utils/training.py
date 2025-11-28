from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

import keras.backend as K
import tensorflow_model_optimization as tfmot
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Optimizer


def regression_loss(y_true, y_pred):
    with K.name_scope('regression_loss'):
        y_true /= 100.0  # because our data is max 100 GeV

        return K.mean(K.square((y_true - y_pred) / y_true), axis=-1)


KerasLossFunction = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class GarNetTrainConfig:
    validation_split: float = 0.25
    batch_size: int = 128
    epochs: int = 120
    prune_epochs: int = 30

    optimizer: Optimizer = field(default_factory=lambda: Adam(learning_rate=0.0005))

    loss_params: Dict[KerasLossFunction, Union[str, KerasLossFunction]] = field(
        default_factory=lambda: {
            'regression': regression_loss,
            'classification': 'binary_crossentropy',
        }
    )

    loss_weight_params: Dict[str, float] = field(default_factory=lambda: {'regression': 0.90, 'classification': 0.10})

    callbacks: List[Callback] = field(
        default_factory=lambda: [
            ReduceLROnPlateau(factor=0.2, patience=10, verbose=1),
            EarlyStopping(verbose=1, patience=10),
        ]
    )

    prune_callbacks: List[Callback] = field(
        default_factory=lambda: [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
    )


garnet_train_config = GarNetTrainConfig()
