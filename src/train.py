import argparse
import os
import random
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from model.hybrid_model import build_hybrid_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid LLM essay detector model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="trained_model",
        help="Base name for saved model artifacts.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="../data/train_embeddings.csv",
        help="Path to embeddings CSV.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="../data/train_features.csv",
        help="Path to stylometric features CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/trained_model",
        help="Directory to save model and plots.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

args = parse_args()
set_seed(args.seed)

print("Loading data...")

# load data
feature_df = pd.read_csv(args.features_path)
embeddings_df = pd.read_csv(args.embeddings_path)


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
    stylo_dropout=0.3,
    stylo_activation="relu",
    dropout= 0.4,
    activation_function="sigmoid"
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
    random_state=args.seed,
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

test_probs = model.predict([X_emb_test, X_stylo_test], verbose=0).ravel()
test_preds = (test_probs >= 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 3)
ConfusionMatrixDisplay(conf_matrix).plot(
    ax=plt.gca(),
    cmap="Blues",
    colorbar=False,
    values_format="d",
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
os.makedirs(args.output_dir, exist_ok=True)

curves_path = os.path.join(args.output_dir, f"{args.model_name}_training_curves.png")
plt.savefig(curves_path, dpi=150)

model_path = os.path.join(args.output_dir, f"{args.model_name}.keras")
model.save(model_path)

report_path = os.path.join(args.output_dir, f"{args.model_name}_training_report.txt")
with open(report_path, "w", encoding="utf-8") as report:
    report.write("Training run report\n")
    report.write("===================\n\n")
    report.write(f"Model name: {args.model_name}\n")
    report.write(f"Seed: {args.seed}\n")
    report.write(f"Embeddings path: {args.embeddings_path}\n")
    report.write(f"Features path: {args.features_path}\n")
    report.write(f"Output dir: {args.output_dir}\n")
    report.write(f"Train size: {len(y_train)}\n")
    report.write(f"Test size: {len(y_test)}\n")
    report.write(f"Class weight: {class_weight}\n")
    report.write(f"Batch size: {history.params.get('batch_size')}\n")
    report.write(f"Epochs: {history.params.get('epochs')}\n")
    report.write(f"Validation split: {history.params.get('validation_split')}\n")
    report.write("\nFinal test metrics\n")
    report.write(f"Test loss: {test_loss:.6f}\n")
    report.write(f"Test accuracy: {test_acc:.6f}\n")

    report.write("\nPer-epoch metrics\n")
    for i in range(len(history.history.get("loss", []))):
        epoch = i + 1
        loss = history.history["loss"][i]
        acc = history.history.get("accuracy", [None])[i]
        val_loss = history.history.get("val_loss", [None])[i]
        val_acc = history.history.get("val_accuracy", [None])[i]

        report.write(
            f"Epoch {epoch:02d}: "
            f"loss={loss:.6f} "
            f"acc={acc:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_acc={val_acc:.6f}\n"
        )

print("Saved model, training curves, and report.")