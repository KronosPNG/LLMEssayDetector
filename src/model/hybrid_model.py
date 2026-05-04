import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_stylometric_branch(
        input_dim: int, 
        output_dim: int = 128,
        dropout: float = 0.3,
        activation_function: str = "relu",
        fc_units: int = 256,
        use_dropout: bool = True,
        use_batch_norm: bool = True,
        shallow_network: bool = False
        ) -> keras.Model:
    """
    Accepts a flat feature vector [batch, input_dim],
    projects it to [batch, output_dim].
    """
    inputs = keras.Input(shape=(input_dim,), name="stylometric_input")

    # First dense layer
    x = layers.Dense(fc_units, name="stylo_dense_1")(inputs)

    if use_batch_norm:
        x = layers.BatchNormalization(name="stylo_bn_1")(x)

    x = layers.Activation(activation_function)(x)

    if use_dropout:
        x = layers.Dropout(dropout)(x)

    if not shallow_network:
        # Second dense layer
        x = layers.Dense(output_dim, name="stylo_dense_2")(x)

        if use_batch_norm:
            x = layers.BatchNormalization(name="stylo_bn_2")(x)

        x = layers.Activation(activation_function)(x)

    return keras.Model(inputs, x, name="stylometric_branch")

def build_hybrid_model(
    n_features: int = 42,
    stylo_out_dim: int = 128,
    stylo_dropout: float = 0.3,
    stylo_fc_units: int = 256,
    stylo_activation: str = "relu",
    embedding_dim: int = 768,
    fc_units: int = 256,
    dropout: float = 0.3,
    activation_function: str = "sigmoid",
    use_dropout: bool = True,
    use_embeddings: bool = True,
    use_stylo: bool = True,
    use_batch_norm: bool = True,
    shallow_stylo: bool = False,
    shallow_classifier: bool = False
) -> keras.Model:

    if use_embeddings:
        # inputs
        emb_input = keras.Input(shape=(embedding_dim,), name="embedding_input") # [B, 768]
        
        if not use_stylo:
            fused = emb_input
    

    if use_stylo:
        # inputs
        stylo_input = keras.Input(shape=(n_features,), name="stylometric_input")

        # Stylometric branch
        stylo_branch = build_stylometric_branch(n_features, stylo_out_dim, stylo_dropout, stylo_activation, stylo_fc_units, use_dropout, use_batch_norm, shallow_stylo)
        f = stylo_branch(stylo_input)   # [B, 128]

        if not use_embeddings:
            fused = f

    if use_embeddings and use_stylo:
        # Fusion
        fused = layers.Concatenate(name="fusion")([emb_input, f])  # [B, 768+128]



    # ** Classifier **
    # First classifier layer
    x = layers.Dense(fc_units, activation="relu")(fused)

    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    if use_dropout:
        x = layers.Dropout(dropout)(x)

    if not shallow_classifier:
        # Second classifier layer
        x = layers.Dense(fc_units // 2, activation="relu")(x)

        if use_dropout:
            x = layers.Dropout(dropout)(x)

    # Output layer
    output = layers.Dense(1, activation=activation_function, name="output")(x)


    inputs = []
    if use_embeddings:
        inputs.append(emb_input)
    if use_stylo:
        inputs.append(stylo_input)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name="hybrid_scibert_stylo"
    )

    return model