import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.hybrid_model import build_hybrid_model

print("Loading data...")

# load data
feature_df = pd.read_csv("../data/train_features.csv")
embeddings_df = pd.read_csv("../data/train_embeddings.csv")




print("Checking id alignment between feature and embedding dataframes...")

# check that every row label in X_emb corresponds to the same row label in X_stylo
if(feature_df["label"].tolist() != embeddings_df["label"].tolist()):
    raise ValueError("Row labels do not match between feature and embedding dataframes.")

y = feature_df["label"].values
X_emb = embeddings_df.drop(columns=["label"]).values
X_stylo = feature_df.drop(columns=["label"]).values

print("Data loaded successfully. Building model...")

#  build
model = build_hybrid_model(
    n_features = 42,
    embedding_dim = 768,
    dropout= 0.4
)

print("Model built successfully. Starting training...")

# compile

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully. Splitting train/test...")

X_emb_train, X_emb_test, X_stylo_train, X_stylo_test, y_train, y_test = train_test_split(
    X_emb,
    X_stylo,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Starting training...")

n0 = np.sum(y_train == 0)
n1 = np.sum(y_train == 1)

class_weight = {
    0: 1.0,
    1: n0 / n1,
}

history = model.fit(
    [X_emb_train, X_stylo_train],
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    shuffle=True,
    class_weight=class_weight
)

test_loss, test_acc = model.evaluate([X_emb_test, X_stylo_test], y_test, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("../data/trained_model/training_curves.png", dpi=150)

model.save("../data/trained_model/trained_model.keras")
print("Saved model and training curves.")