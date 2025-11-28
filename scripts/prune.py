import argparse
from math import ceil

import tensorflow_model_optimization as tfmot
from keras.layers import Dense
from keras.models import clone_model, load_model
from qkeras.utils import _add_supported_quantized_objects

from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer
from utils.data import load_data
from utils.files import model_path
from utils.training import garnet_train_config as gtc
from utils.training import garnet_train_config as train_config
from utils.training import regression_loss


def apply_pruning(layer, end_step):
    if isinstance(layer, Dense):
        if layer.name in ['q_dense', 'q_dense_1', 'q_dense_3']:
            final_sparsity = 0.5
        elif layer.name in [
            'q_dense_2',
        ]:
            final_sparsity = 0.75
        else:
            return layer
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=final_sparsity,
                begin_step=0,
                end_step=end_step,
                frequency=500,
            )
        }
        print(f'Pruning layer {layer.name} with PolynomialDecay, final sparsity = {0.75}')
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

    return layer


def parse_args():
    parser = argparse.ArgumentParser(
        prog='PruneGarnet',
        description='Prune an already trained GarNet model',
    )
    parser.add_argument('-i', '--model_infile')
    parser.add_argument('-o', '--model_outfile')
    parser.add_argument('-p', '--precision', nargs='+', type=int)
    parser.add_argument('-s', '--dataset_suffix', default='baseline_50')

    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, _, _ = load_data(suffix=args.dataset_suffix)

    # We do not need the cluster sizes
    X_train = X_train[0]

    co = {}
    _add_supported_quantized_objects(co)
    co['GarNetLayer'] = GarNetLayer
    co['regression_loss'] = regression_loss
    pretrained_model = load_model(str(model_path / args.model_infile), custom_objects=co)

    steps_per_epoch = ceil(len(X_train) / train_config.batch_size)
    epochs = train_config.prune_epochs
    end_step = steps_per_epoch * epochs
    print(f'Pruning end_step set to {end_step} (steps_per_epoch {steps_per_epoch} * epochs {epochs})')

    model_for_pruning = clone_model(
        pretrained_model,
        clone_function=lambda layer: apply_pruning(layer, end_step),
    )
    model_for_pruning.compile(optimizer=gtc.optimizer, loss=gtc.loss_params, loss_weights=gtc.loss_weight_params)

    print('Pruning model...')
    model_for_pruning.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        callbacks=train_config.prune_callbacks,
        validation_split=train_config.validation_split,
        batch_size=train_config.batch_size,
        shuffle=True,
        verbose=1,
    )

    print('Retraining pruned model')
    model_for_pruning.compile(
        optimizer=train_config.optimizer,
        loss=train_config.loss_params,
        loss_weights=train_config.loss_weight_params,
    )
    model_for_pruning.fit(
        X_train,
        y_train,
        epochs=train_config.epochs,
        initial_epoch=train_config.prune_epochs,
        callbacks=train_config.callbacks,
        validation_split=train_config.validation_split,
        batch_size=train_config.batch_size,
        shuffle=True,
        verbose=1,
    )

    # Reload model and strip pruning
    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    final_model.save(str(model_path / args.model_outfile))  # Overwrite un-stripped model


if __name__ == '__main__':
    main()
