import argparse
import os
import random
import pandas as pd
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model.hybrid_model import build_hybrid_model
from utils.genetic_algorithm import GeneticAlgorithmOptimizer

# Hyperparameter space (same as in optimisation.py)
HYPERPARAMETER_SPACE = {
    "stylo_dropout": [0.0, 0.1, 0.25, 0.4, 0.5],
    "stylo_activation": ["relu", "tanh", "leaky_relu", "sigmoid"],
    "dropout": [0.0, 0.1, 0.25, 0.4, 0.5],
    "activation_function": ["relu", "tanh", "leaky_relu", "sigmoid"],
    "optimizer": [keras.optimizers.Adam, keras.optimizers.RMSprop, keras.optimizers.SGD],
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "stylo_fc_units": [128, 256, 512],
    "fc_units": [128, 256, 512],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetic algorithm hyperparameter optimization.")
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
        default="../data/ga_results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size for genetic algorithm.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Mutation rate (0-1).",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Crossover rate (0-1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs per fitness evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split size (from training data).",
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


def create_fitness_function(X_emb_train, X_stylo_train, y_train, X_emb_val, 
                            X_stylo_val, y_val, epochs):
    """
    Create a fitness function that trains and evaluates a model.
    
    Args:
        X_emb_train: Training embeddings
        X_stylo_train: Training stylometric features
        y_train: Training labels
        X_emb_val: Validation embeddings
        X_stylo_val: Validation stylometric features
        y_val: Validation labels
        epochs: Number of epochs to train
        
    Returns:
        Fitness function
    """
    def fitness_function(hyperparams):
        try:
            keras.backend.clear_session()
            
            # Build and compile model with hyperparameters
            model = build_hybrid_model(
                stylo_dropout=hyperparams["stylo_dropout"],
                stylo_activation=hyperparams["stylo_activation"],
                dropout=hyperparams["dropout"],
                activation_function=hyperparams["activation_function"],
            )
            
            optimizer_class = hyperparams["optimizer"]
            optimizer = optimizer_class(learning_rate=hyperparams["learning_rate"])
            
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            
            # Calculate class weights
            n0 = np.sum(y_train == 0)
            n1 = np.sum(y_train == 1)
            class_weight = {0: 1.0, 1: n0 / n1} if n0 > 0 and n1 > 0 else None
            
            # Train model
            history = model.fit(
                [X_emb_train, X_stylo_train],
                y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=([X_emb_val, X_stylo_val], y_val),
                class_weight=class_weight,
                verbose=0,
            )
            
            # Return best validation accuracy as fitness
            best_val_acc = max(history.history["val_accuracy"])
            return best_val_acc
            
        except Exception as e:
            print(f"Error evaluating hyperparameters: {e}")
            return 0.0
    
    return fitness_function


args = parse_args()
set_seed(args.seed)

print("Loading data...")
feature_df = pd.read_csv(args.features_path)
embeddings_df = pd.read_csv(args.embeddings_path)

print("Checking data alignment...")
if feature_df["label"].tolist() != embeddings_df["label"].tolist():
    raise ValueError("Row labels do not match between feature and embedding dataframes.")

y = feature_df["label"].values
X_emb = embeddings_df.drop(columns=["label"]).values
X_stylo = feature_df.drop(columns=["label"]).values

print("Splitting data...")
X_emb_train, X_emb_test, X_stylo_train, X_stylo_test, y_train, y_test = train_test_split(
    X_emb, X_stylo, y,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=y,
)

X_emb_train, X_emb_val, X_stylo_train, X_stylo_val, y_train, y_val = train_test_split(
    X_emb_train, X_stylo_train, y_train,
    test_size=args.val_size,
    random_state=args.seed,
    stratify=y_train,
)

print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")

# Create fitness function
fitness_function = create_fitness_function(
    X_emb_train, X_stylo_train, y_train,
    X_emb_val, X_stylo_val, y_val,
    epochs=args.epochs
)

# Initialize and run genetic algorithm
print("\nStarting genetic algorithm optimization...")
ga = GeneticAlgorithmOptimizer(
    hyperparameter_space=HYPERPARAMETER_SPACE,
    population_size=args.population_size,
    generations=args.generations,
    mutation_rate=args.mutation_rate,
    crossover_rate=args.crossover_rate,
    elite_ratio=0.1,
    random_seed=args.seed,
)

best_individual = ga.optimize(fitness_function)

print(f"\n{'='*60}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Best fitness: {best_individual.fitness:.6f}")
print(f"\nBest hyperparameters found:")
for param_name, param_value in best_individual.hyperparameters.items():
    if isinstance(param_value, type) and issubclass(param_value, keras.optimizers.Optimizer):
        print(f"  {param_name}: {param_value.__name__}")
    else:
        print(f"  {param_name}: {param_value}")

# Evaluate on test set
print("\nEvaluating best model on test set...")
keras.backend.clear_session()

best_model = build_hybrid_model(
    stylo_dropout=best_individual.hyperparameters["stylo_dropout"],
    stylo_activation=best_individual.hyperparameters["stylo_activation"],
    dropout=best_individual.hyperparameters["dropout"],
    activation_function=best_individual.hyperparameters["activation_function"],
)

optimizer_class = best_individual.hyperparameters["optimizer"]
optimizer = optimizer_class(learning_rate=best_individual.hyperparameters["learning_rate"])

best_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

n0 = np.sum(y_train == 0)
n1 = np.sum(y_train == 1)
class_weight = {0: 1.0, 1: n0 / n1} if n0 > 0 and n1 > 0 else None

best_model.fit(
    [X_emb_train, X_stylo_train],
    y_train,
    batch_size=32,
    epochs=args.epochs,
    class_weight=class_weight,
    verbose=0,
)

test_loss, test_acc = best_model.evaluate([X_emb_test, X_stylo_test], y_test, verbose=0)
print(f"Test loss: {test_loss:.6f}")
print(f"Test accuracy: {test_acc:.6f}")

# Save results
os.makedirs(args.output_dir, exist_ok=True)

# Save history as CSV
history_df = pd.DataFrame(ga.get_history())
history_path = os.path.join(args.output_dir, "ga_optimization_history.csv")
history_df.to_csv(history_path, index=False)

# Plot optimization progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_df["generation"], history_df["best_fitness"], marker='o', label="Best")
plt.plot(history_df["generation"], history_df["avg_fitness"], marker='s', label="Average")
plt.xlabel("Generation")
plt.ylabel("Fitness (Validation Accuracy)")
plt.title("GA Optimization Progress")
plt.legend()
plt.grid(True, alpha=0.3)

# Save best hyperparameters
best_params_path = os.path.join(args.output_dir, "best_hyperparameters.txt")
with open(best_params_path, "w") as f:
    f.write("Best Hyperparameters Found\n")
    f.write("="*50 + "\n")
    f.write(f"Best fitness (val accuracy): {best_individual.fitness:.6f}\n")
    f.write(f"Test accuracy: {test_acc:.6f}\n")
    f.write(f"Test loss: {test_loss:.6f}\n\n")
    f.write("Hyperparameters:\n")
    for param_name, param_value in best_individual.hyperparameters.items():
        if isinstance(param_value, type) and issubclass(param_value, keras.optimizers.Optimizer):
            f.write(f"  {param_name}: {param_value.__name__}\n")
        else:
            f.write(f"  {param_name}: {param_value}\n")

plt.tight_layout()
plot_path = os.path.join(args.output_dir, "ga_progress.png")
plt.savefig(plot_path, dpi=150)
print(f"\nResults saved to {args.output_dir}")
