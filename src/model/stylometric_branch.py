import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_stylometric_branch(input_dim: int, output_dim: int = 128) -> keras.Model:
    """
    Accepts a flat feature vector [batch, input_dim],
    projects it to [batch, output_dim].
    """
    inputs = keras.Input(shape=(input_dim,), name="stylometric_input")

    x = layers.Dense(256, name="stylo_dense_1")(inputs)
    x = layers.BatchNormalization(name="stylo_bn_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(output_dim, name="stylo_dense_2")(x)
    x = layers.BatchNormalization(name="stylo_bn_2")(x)
    x = layers.Activation("relu")(x)

    return keras.Model(inputs, x, name="stylometric_branch")