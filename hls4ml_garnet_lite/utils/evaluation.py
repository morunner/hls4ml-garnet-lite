from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import Model
from sklearn.metrics import roc_auc_score, roc_curve


def garnet_predict(model: Model, x_test: list) -> Tuple[list, list]:
    y_pred = model.predict(x_test)
    y_energy_pred, y_pid_pred = y_pred[0], y_pred[1]

    return y_energy_pred, y_pid_pred


def response_rmse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    response = tf.math.divide_no_nan(y_pred, y_true)
    return tf.sqrt(tf.reduce_mean(tf.square(response - 1.0)))


def compare_keras_hls_predictions(
    test_energy_pred: np.ndarray,
    test_pid_pred: np.ndarray,
    test_energy_pred_hls: np.ndarray,
    test_pid_pred_hls: np.ndarray,
    test_energy_true: np.ndarray,
    test_pid_true: np.ndarray,
):
    test_response_rmse = response_rmse(test_energy_true, test_energy_pred)
    test_auc = roc_auc_score(test_pid_true, test_pid_pred)
    fpr, tpr, _ = roc_curve(test_pid_true, test_pid_pred)

    hls_response_rmse = response_rmse(test_energy_true, test_energy_pred_hls)
    hls_auc = roc_auc_score(test_pid_true, test_pid_pred_hls)
    hls_fpr, hls_tpr, _ = roc_curve(test_pid_true, test_pid_pred_hls)

    with sns.axes_style('darkgrid'):
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.plot(1 - fpr, tpr, label=f'Keras (AUC: {test_auc:.3f})')
        plt.plot(1 - hls_fpr, hls_tpr, label=f'HLS (AUC: {hls_auc:.3f})')

        plt.xlabel('Pion False Positive Rate')
        plt.ylabel('Pion True Positive Rate')
        plt.xlim(0.7, 1.0)
        plt.ylim(0.7, 1.0)
        plt.title('Pion Identification ROC Curve')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)

        plt.hist(
            test_energy_pred.flatten() / test_energy_true,
            bins=50,
            histtype='stepfilled',
            alpha=0.5,
            density=True,
            label=f'Keras (RMSE: {test_response_rmse:.3f})',
        )

        plt.hist(
            test_energy_pred_hls.flatten() / test_energy_true,
            bins=50,
            histtype='stepfilled',
            alpha=0.5,
            density=True,
            label=f'HLS (RMSE: {hls_response_rmse:.3f})',
        )

        plt.axvline(1.0, color='k', linestyle='--', lw=1, alpha=0.7)
        plt.xlabel('Predicted / True Energy')
        plt.ylabel('Density')
        plt.xlim(0.0, 2.0)
        plt.title('Energy Response Distribution')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
