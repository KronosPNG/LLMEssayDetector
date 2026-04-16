import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_stylometric_branch(
        input_dim: int, 
        output_dim: int = 128,
        dropout: float = 0.3,
        activation_function: str = "relu"
        ) -> keras.Model:
    """
    Accepts a flat feature vector [batch, input_dim],
    projects it to [batch, output_dim].
    """
    inputs = keras.Input(shape=(input_dim,), name="stylometric_input")

    x = layers.Dense(256, name="stylo_dense_1")(inputs)
    x = layers.BatchNormalization(name="stylo_bn_1")(x)
    x = layers.Activation(activation_function)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(output_dim, name="stylo_dense_2")(x)
    x = layers.BatchNormalization(name="stylo_bn_2")(x)
    x = layers.Activation(activation_function)(x)

    return keras.Model(inputs, x, name="stylometric_branch")

def build_hybrid_model(
    n_features: int = 42,
    stylo_out_dim: int = 128,
    stylo_dropout: float = 0.3,
    stylo_activation: str = "relu",
    embedding_dim: int = 768,
    fc_units: int = 256,
    dropout: float = 0.3,
    activation_function: str = "sigmoid"
) -> keras.Model:

    # Inputs
    emb_input   = keras.Input(shape=(embedding_dim,), name="embedding_input")
    stylo_input = keras.Input(shape=(n_features,), name="stylometric_input")

    # Stylometric branch
    stylo_branch = build_stylometric_branch(n_features, stylo_out_dim, stylo_dropout, stylo_activation)
    f = stylo_branch(stylo_input)   # [B, 128]

    # Fusion
    fused = layers.Concatenate(name="fusion")([emb_input, f])  # [B, 768 + 128]

    # Classifier
    x = layers.Dense(fc_units, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(fc_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    output = layers.Dense(1, activation=activation_function, name="output")(x)

    model = keras.Model(
        inputs=[emb_input, stylo_input],
        outputs=output,
        name="hybrid_scibert_stylo"
    )

    return model