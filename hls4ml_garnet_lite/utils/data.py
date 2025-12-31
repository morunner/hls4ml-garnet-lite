import os
import urllib
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from hls4ml_garnet_lite.utils.files import dataset_path


def fetch_garnet_dataset(n_files: int) -> None:
    # From https://zenodo.org/records/3888910
    for i in range(n_files):
        events_url = f'https://zenodo.org/records/3888910/files/events_{i}.h5?download=1'
        events_filename = f'events_{i}.h5'

        if not os.path.exists(dataset_path / events_filename):
            urllib.request.urlretrieve(events_url, dataset_path / events_filename)


def split_data_and_save(
    n_files: int,
    suffix: str = None,
) -> None:
    with h5py.File(dataset_path / 'events_0.h5', 'r') as init_data:
        X_hits, X_size = init_data['cluster'][:, :], init_data['size'][:]
        y_energy, y_pid = init_data['truth_energy'][:], init_data['truth_pid'][:]

    for i in range(1, n_files):
        try:
            with h5py.File(dataset_path / f'events_{i}.h5', 'r') as init_data:
                X_hits = np.concatenate((X_hits, init_data['cluster'][:, :]))
                X_size = np.concatenate((X_size, init_data['size'][:]))
                y_energy = np.concatenate((y_energy, init_data['truth_energy'][:]))
                y_pid = np.concatenate((y_pid, init_data['truth_pid'][:]))
        except OSError:
            print(f'Unable to open file events_{i}.h5')

    (
        X_hits_train,
        X_hits_test,
        X_size_train,
        X_size_test,
        y_energy_train,
        y_energy_test,
        y_pid_train,
        y_pid_test,
    ) = train_test_split(X_hits, X_size, y_energy, y_pid)

    suffix = f'_{suffix}' if suffix is not None else ''
    np.save(dataset_path / f'X_hits_train{suffix}.npy', np.array(X_hits_train))
    np.save(dataset_path / f'X_size_train{suffix}.npy', np.array(X_size_train))
    np.save(dataset_path / f'y_energy_train{suffix}.npy', np.array(y_energy_train))
    np.save(dataset_path / f'y_pid_train{suffix}.npy', np.array(y_pid_train))
    np.save(dataset_path / f'X_hits_test{suffix}.npy', np.array(X_hits_test))
    np.save(dataset_path / f'X_size_test{suffix}.npy', np.array(X_size_test))
    np.save(dataset_path / f'y_energy_test{suffix}.npy', np.array(y_energy_test))
    np.save(dataset_path / f'y_pid_test{suffix}.npy', np.array(y_pid_test))


def load_data(base_path: Path = dataset_path, suffix: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    suffix = f'_{suffix}' if suffix is not None else ''

    X_hits_train = np.load(base_path / f'X_hits_train{suffix}.npy')
    X_size_train = np.load(base_path / f'X_size_train{suffix}.npy')
    y_energy_train = np.load(base_path / f'y_energy_train{suffix}.npy')
    y_pid_train = np.load(base_path / f'y_pid_train{suffix}.npy')
    X_hits_test = np.load(base_path / f'X_hits_test{suffix}.npy')
    X_size_test = np.load(base_path / f'X_size_test{suffix}.npy')
    y_energy_test = np.load(base_path / f'y_energy_test{suffix}.npy')
    y_pid_test = np.load(base_path / f'y_pid_test{suffix}.npy')

    return (
        [X_hits_train, X_size_train],
        [y_energy_train, y_pid_train],
        [X_hits_test, X_size_test],
        [y_energy_test, y_pid_test],
    )
