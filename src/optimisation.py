import argparse
import os
import random
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from model.hybrid_model import build_hybrid_model

HYPERPARAMETERS = {
    "stylo_dropout": [0.0 , 0.1 , 0.25 , 0.4 , 0.5],
    "stylo_activation": ["relu", "tanh", "leaky_relu", "sigmoid"],
    "dropout": [0.0 , 0.1 , 0.25 , 0.4 , 0.5],
    "activation_function": ["relu", "tanh", "leaky_relu", "sigmoid"],
    "optimizer": [keras.optimizers.Adam, keras.optimizers.RMSprop, keras.optimizers.SGD],
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "stylo_fc_units": [128, 256, 512],
    "fc_units": [128, 256, 512],
}