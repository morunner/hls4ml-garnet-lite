import numpy as np
import pandas as pd
from keras.models import Model, load_model
from qkeras.utils import _add_supported_quantized_objects
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from utils.data import load_data
from utils.evaluation import response_rmse
from utils.files import model_path
from utils.training import regression_loss

from hls4ml.contrib.garnet import GarNet
from hls4ml_garnet_lite.keras_model.garnet_lite import GarNetLayer

MODELS = [
    'original_garnet_8_0',
    'original_garnet_lite_8_0',
    'garnet_8_0',
    'garnet_8_0_pruned',
    'garnet_lite_8_0',
    'garnet_lite_8_0_pruned',
]


def add_model_predictions_to_df(
    keras_model: Model, X_test: np.ndarray, y_test: np.ndarray, df: pd.DataFrame, name: str
) -> pd.DataFrame:
    test_energy_pred, test_pid_pred = keras_model.predict(X_test)
    test_energy_pred *= 100

    test_response_rmse = response_rmse(y_test[0], test_energy_pred)
    test_auc = roc_auc_score(y_test[1], test_pid_pred)

    row = pd.Series(
        {
            'model': name,
            'auc': test_auc,
            'rmse': test_response_rmse,
        }
    )
    df = pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)
    return df


def main():
    _, _, X_test, y_test = load_data(suffix='baseline_50')

    co = {}
    _add_supported_quantized_objects(co)
    co['GarNet'] = GarNet
    co['GarNetLayer'] = GarNetLayer
    X_test = X_test[0]
    co['regression_loss'] = regression_loss

    df = pd.DataFrame(columns=['model', 'auc', 'rmse'])
    for model in MODELS:
        keras_model_path = str(model_path / f'{model}.keras')
        keras_model = load_model(keras_model_path, custom_objects=co)
        if 'original' in model:
            df = add_model_predictions_to_df(keras_model, X_test, y_test, df, model)
        else:
            df = add_model_predictions_to_df(keras_model, X_test[0], y_test, df, model)

    # Print metrics
    print(tabulate(df.round(3), headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main()
