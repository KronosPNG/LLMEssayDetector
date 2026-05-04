import argparse
import os
import random
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from model.hybrid_model import build_hybrid_model

ABLATION_MODES = {
    "full": "Full model (baseline)",
    "no_embeddings": "Full model without embeddings branch",
    "no_stylo": "Full model without stylometric branch",
    "no_dropout": "Full model without dropout",
    "no_batch_norm": "Full model without batch normalization",
    "shallow_stylo": "Full model with a shallower stylometric branch (1 dense layer instead of 2)",
    "shallow_classifier": "Full model with a shallower classifier branch (1 dense layer instead of 2)",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study for the hybrid LLM essay detector.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="../data/processed_datasets/train_embeddings.csv",
        help="Path to embeddings CSV.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="../data/processed_datasets/train_features.csv",
        help="Path to stylometric features CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/ablation_study",
        help="Directory to save results of the ablation study.",
    )
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=ABLATION_MODES.keys(),
        default="full",
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


model_name = f"ablation_{args.ablation_type}"
output_dir = os.path.join(args.output_dir, model_name)

print("Loading data...")

use_embeddings = True
use_stylo = True
use_dropout = True
use_batch_norm = True
shallow_stylo = False
shallow_classifier = False
# set variables depending on ablation type

if args.ablation_type == "no_embeddings":
    use_embeddings = False

elif args.ablation_type == "no_stylo":
    use_stylo = False

elif args.ablation_type == "no_dropout":
    use_dropout = False

elif args.ablation_type == "no_batch_norm":
    use_batch_norm = False

elif args.ablation_type == "shallow_stylo":
    shallow_stylo = True

elif args.ablation_type == "shallow_classifier":
    shallow_classifier = True

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

def build_compiled_model() -> keras.Model:
    keras.backend.clear_session()

    model = build_hybrid_model(
        stylo_dropout=0.3,
        stylo_activation="relu",
        dropout=0.4,
        activation_function="sigmoid",
        use_embeddings=use_embeddings,
        use_stylo=use_stylo,
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        shallow_stylo=shallow_stylo,
        shallow_classifier=shallow_classifier
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

print("Model factory ready. Training on the full dataset...")

n0 = np.sum(y == 0)
n1 = np.sum(y == 1)
class_weight = None
if n0 > 0 and n1 > 0:
    class_weight = {
        0: 1.0,
        1: n0 / n1,
    }

model = build_compiled_model()

training_inputs = []
if use_embeddings:
    training_inputs.append(X_emb)
if use_stylo:    
    training_inputs.append(X_stylo)

history = model.fit(
    training_inputs,
    y,
    batch_size=32,
    epochs=10,
    shuffle=True,
    class_weight=class_weight,
    verbose=1,
)

evaluation_inputs = []
if use_embeddings:
    evaluation_inputs.append(X_emb)
if use_stylo:
    evaluation_inputs.append(X_stylo)

train_loss, train_acc = model.evaluate(evaluation_inputs, y, verbose=0)
print(f"Training loss: {train_loss:.4f} | Training accuracy: {train_acc:.4f}")

train_probs = model.predict(evaluation_inputs, verbose=0).ravel()
train_preds = (train_probs >= 0.5).astype(int)
conf_matrix = confusion_matrix(y, train_preds)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history["accuracy"], label="train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history["loss"], label="train")
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
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

curves_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
plt.savefig(curves_path, dpi=150)

model_path = os.path.join(output_dir, f"{model_name}.keras")
model.save(model_path)

report_path = os.path.join(output_dir, f"{model_name}_training_report.txt")
with open(report_path, "w", encoding="utf-8") as report:
    report.write("Training run report\n")
    report.write("===================\n\n")
    report.write(f"Model name: {model_name}\n")
    report.write("\nFinal dataset metrics\n")
    report.write(f"Dataset loss: {train_loss:.6f}\n")
    report.write(f"Dataset accuracy: {train_acc:.6f}\n")

    report.write("\nPer-epoch metrics\n")
    for i in range(len(history.history.get("loss", []))):
        epoch = i + 1
        loss = history.history["loss"][i]
        acc = history.history.get("accuracy", [None])[i]

        report.write(
            f"Epoch {epoch:02d}: "
            f"loss={loss:.6f} "
            f"acc={acc:.6f}\n"
        )

print("Saved model, training curves, and report.")