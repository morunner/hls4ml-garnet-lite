#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np
import h5py
from argparse import ArgumentParser

QUANTIZED = True
VMAX = 128

parser = ArgumentParser(description='Train the keras model.')
parser.add_argument('train_path', metavar='PATH', help='Training data file.')
parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', help='Validation data file.')
parser.add_argument('--weights', '-w', metavar='PATH', dest='weights_path', help='Read weights from HDF5.')
parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDF5 output to path.')
parser.add_argument('--ngpu', '-j', metavar='N', dest='ngpu', type=int, default=1, help='Use N GPUs.')
parser.add_argument('--gpus', '-g', metavar='ID', dest='gpus', type=int, nargs='+', help='Use specified GPUs (overrides --ngpu).')
parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')

args = parser.parse_args()
del sys.argv[1:]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join('%d' % u for u in args.gpus)
    args.ngpu = len(args.gpus)

try:
    import tensorflow.keras as keras
except ImportError:
    import keras
import gpu_hack

from model import make_model, regression_loss, classification_loss

model = make_model(VMAX, QUANTIZED)

if args.weights_path is not None:
    model.load_weights(args.weights_path)

model_single = model
if args.ngpu > 1:
    model = keras.utils.multi_gpu_model(model_single, args.ngpu)

optimizer = keras.optimizers.Adam(lr=0.0005)

losses = {'regression': regression_loss, 'classification': classification_loss}
loss_weights = {'regression': 0.99, 'classification': 0.01}
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(verbose=1, patience=10)
]

fit_args = {'epochs': args.num_epochs, 'callbacks': callbacks, 'shuffle': True}

with h5py.File(args.train_path, 'r') as data_source:
    x_hits = data_source['cluster'][:, :VMAX]
    x_size = data_source['size'][:]
    y_energy = data_source['truth_energy'][:]
    y_pid = data_source['truth_pid'][:]

fit_args['x'] = [x_hits, x_size]
fit_args['y'] = {'regression': y_energy, 'classification': y_pid}

if args.validation_path:
    with h5py.File(args.validation_path, 'r') as data_source:
        valid_x_hits = data_source['cluster'][:, :VMAX]
        valid_x_size = data_source['size'][:]
        valid_y_energy = data_source['truth_energy'][:]
        valid_y_pid = data_source['truth_pid'][:]

    fit_args['validation_data'] = ([valid_x_hits, valid_x_size], {'regression': valid_y_energy, 'classification': valid_y_pid})

try:
    model.fit(**fit_args)
except KeyboardInterrupt:
    pass

if args.out_path:
    model_single.save(args.out_path)
