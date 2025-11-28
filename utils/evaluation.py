from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from sklearn.metrics import root_mean_squared_error


def garnet_predict(model: Model, x_test: list) -> Tuple[list, list]:
    y_pred = model.predict(x_test)
    y_energy_pred, y_pid_pred = y_pred[0], y_pred[1]

    return y_energy_pred, y_pid_pred


def split_energy_into_electrons_pions(truth_pid: np.ndarray, y_energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    filter = np.array(list(map(bool, truth_pid)))
    y_energy_e = y_energy[~filter]
    y_energy_p = y_energy[filter]
    return y_energy_e, y_energy_p


def bin_ref_comp_response_by_ref_pred_energies(
    y_energy_ref: np.ndarray, y_energy_comp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(10, 90, 9).astype(int)  # Create 10 GeV bins from 10 to 100 GeV
    indices = np.digitize(y_energy_ref, bins)
    responses = y_energy_comp / y_energy_ref

    output = []
    for i in range(1, len(bins) + 1):
        output.append(responses[indices == i])

    return output, bins


def response_boxplot(
    y_pid_test: np.ndarray,
    y_energy_ref: np.ndarray,
    y_energy_comp: np.ndarray,
    title: str,
    ylabel: str,
):
    y_energy_ref_e, y_energy_ref_p = split_energy_into_electrons_pions(y_pid_test, y_energy_ref)
    y_energy_comp_e, y_energy_comp_p = split_energy_into_electrons_pions(y_pid_test, y_energy_comp)

    y_energy_e_bins, _ = bin_ref_comp_response_by_ref_pred_energies(y_energy_ref_e, y_energy_comp_e)
    y_energy_p_bins, bins = bin_ref_comp_response_by_ref_pred_energies(y_energy_ref_p, y_energy_comp_p)

    rms_tot = root_mean_squared_error(y_energy_comp / y_energy_ref, np.ones(y_energy_ref.shape))
    rms_e = root_mean_squared_error(y_energy_comp_e / y_energy_ref_e, np.ones(y_energy_ref_e.shape))
    rms_p = root_mean_squared_error(y_energy_comp_p / y_energy_ref_p, np.ones(y_energy_ref_p.shape))

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    axs[0].boxplot(y_energy_e_bins, tick_labels=bins, whis=[5, 95], showfliers=False)
    axs[0].set_title(f'Electrons (RMS = {rms_e:.3f})')
    axs[1].boxplot(y_energy_p_bins, tick_labels=bins, whis=[5, 95], showfliers=False)
    axs[1].set_title(f'Pions (RMS = {rms_p:.3f})')

    fig.supxlabel('Primary Particle Energy (GeV)')
    fig.supylabel(ylabel)
    for ax in axs:
        ax.yaxis.grid(True)
        ax.axhline(y=1, alpha=0.4, linestyle='dotted', color='red')

    fig.suptitle(f'{title} (RMS = {rms_tot:.3f})')
    fig.tight_layout()
    fig.show()
