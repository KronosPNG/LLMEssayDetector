import tensorflow as tf
from tensorflow import keras
from keras import layers

from .stylometric_branch import build_stylometric_branch


def build_hybrid_model(
    embedding_dim: int = 768,
    n_features: int = 42,
    stylo_out_dim: int = 128,
    fc_units: int = 256,
    dropout: float = 0.3,
) -> keras.Model:

    # 🔹 Inputs
    emb_input   = keras.Input(shape=(embedding_dim,), name="embedding_input")
    stylo_input = keras.Input(shape=(n_features,), name="stylometric_input")

    # 🔹 Stylometric branch
    stylo_branch = build_stylometric_branch(n_features, stylo_out_dim)
    f = stylo_branch(stylo_input)   # [B, 128]

    # 🔹 Fusion
    fused = layers.Concatenate(name="fusion")([emb_input, f])  # [B, 768 + 128]

    # 🔹 Classifier
    x = layers.Dense(fc_units, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(fc_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(
        inputs=[emb_input, stylo_input],
        outputs=output,
        name="hybrid_scibert_stylo"
    )

    return model