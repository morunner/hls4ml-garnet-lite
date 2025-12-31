import argparse

from keras.callbacks import ModelCheckpoint
from utils.data import load_data
from utils.files import model_path
from utils.training import garnet_train_config as train_config

from hls4ml_garnet_lite.keras_model.garnet_lite import QGarNetFactory


def parse_args():
    parser = argparse.ArgumentParser(prog='TrainGarNet', description='Train a GarNet model on given baseline data')
    parser.add_argument('-m', '--model_file')
    parser.add_argument('-p', '--precision', nargs='+', type=int)
    parser.add_argument('-s', '--dataset_suffix', default='baseline_50')

    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, _, _ = load_data(suffix=args.dataset_suffix)

    # We do not need the cluster sizes
    X_train = X_train[0]

    model = QGarNetFactory(8, 4, 8, precision=args.precision).create_keras_model()
    model.compile(
        optimizer=train_config.optimizer, loss=train_config.loss_params, loss_weights=train_config.loss_weight_params
    )

    callbacks = list(train_config.callbacks)
    callbacks.append(ModelCheckpoint(filepath=str(model_path / args.model_file), save_best_only=True))
    model.fit(
        x=X_train,
        y={'regression': y_train[0], 'classification': y_train[1]},
        epochs=train_config.epochs,
        callbacks=callbacks,
        validation_split=train_config.validation_split,
        batch_size=train_config.batch_size,
        shuffle=True,
        verbose=1,
    )


if __name__ == '__main__':
    main()
