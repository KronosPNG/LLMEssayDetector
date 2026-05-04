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
        default="../data/trained_model",
        help="Directory to save model and plots.",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per fold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--select-metric",
        type=str,
        default="val_accuracy",
        choices=["val_accuracy", "val_loss"],
        help="Metric used to select the best fold model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test split size.",
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

def build_compiled_model() -> keras.Model:
    keras.backend.clear_session()
    model = build_hybrid_model(
        stylo_dropout=0.3,
        stylo_activation="relu",
        dropout=0.4,
        activation_function="sigmoid",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

print("Model factory ready. Splitting train/test...")

X_emb_train, X_emb_test, X_stylo_train, X_stylo_test, y_train, y_test = train_test_split(
    X_emb,
    X_stylo,
    y,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=y,
)

print("Starting K-fold training...")

skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
best_metric = None
best_fold = None
best_history = None
best_model = None
fold_summaries = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_emb_train, y_train), start=1):
    X_emb_fold_train = X_emb_train[train_idx]
    X_stylo_fold_train = X_stylo_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_emb_fold_val = X_emb_train[val_idx]
    X_stylo_fold_val = X_stylo_train[val_idx]
    y_fold_val = y_train[val_idx]

    n0 = np.sum(y_fold_train == 0)
    n1 = np.sum(y_fold_train == 1)
    class_weight = {
        0: 1.0,
        1: n0 / n1,
    }

    print(f"Fold {fold_idx}/{args.kfolds}: training...")
    model = build_compiled_model()
    history = model.fit(
        [X_emb_fold_train, X_stylo_fold_train],
        y_fold_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=([X_emb_fold_val, X_stylo_fold_val], y_fold_val),
        shuffle=True,
        class_weight=class_weight,
        verbose=1,
    )

    if args.select_metric == "val_loss":
        fold_metric = float(np.min(history.history["val_loss"]))
        is_better = best_metric is None or fold_metric < best_metric
    else:
        fold_metric = float(np.max(history.history["val_accuracy"]))
        is_better = best_metric is None or fold_metric > best_metric

    fold_summaries.append(
        {
            "fold": fold_idx,
            "metric": fold_metric,
            "class_weight": class_weight,
        }
    )

    if is_better:
        best_metric = fold_metric
        best_fold = fold_idx
        best_history = history
        best_model = model

print(f"Best fold: {best_fold} | {args.select_metric}={best_metric:.6f}")

if best_model is None or best_history is None:
    raise RuntimeError("No model selected from cross-validation.")

test_loss, test_acc = best_model.evaluate([X_emb_test, X_stylo_test], y_test, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

test_probs = best_model.predict([X_emb_test, X_stylo_test], verbose=0).ravel()
test_preds = (test_probs >= 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(best_history.history["accuracy"], label="train")
plt.plot(best_history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(best_history.history["loss"], label="train")
plt.plot(best_history.history["val_loss"], label="val")
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
best_model.save(model_path)

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
    report.write(f"K-folds: {args.kfolds}\n")
    report.write(f"Selection metric: {args.select_metric}\n")
    report.write(f"Best fold: {best_fold}\n")
    report.write(f"Best fold metric: {best_metric:.6f}\n")
    report.write(f"Batch size: {args.batch_size}\n")
    report.write(f"Epochs: {args.epochs}\n")
    report.write("\nFinal test metrics\n")
    report.write(f"Test loss: {test_loss:.6f}\n")
    report.write(f"Test accuracy: {test_acc:.6f}\n")

    report.write("\nPer-epoch metrics\n")
    report.write("\nFold summaries\n")
    for summary in fold_summaries:
        report.write(
            f"Fold {summary['fold']}: metric={summary['metric']:.6f} "
            f"class_weight={summary['class_weight']}\n"
        )

    report.write("\nPer-epoch metrics (best fold)\n")
    for i in range(len(best_history.history.get("loss", []))):
        epoch = i + 1
        loss = best_history.history["loss"][i]
        acc = best_history.history.get("accuracy", [None])[i]
        val_loss = best_history.history.get("val_loss", [None])[i]
        val_acc = best_history.history.get("val_accuracy", [None])[i]

        report.write(
            f"Epoch {epoch:02d}: "
            f"loss={loss:.6f} "
            f"acc={acc:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_acc={val_acc:.6f}\n"
        )

print("Saved model, training curves, and report.")