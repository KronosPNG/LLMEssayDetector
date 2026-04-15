import argparse
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from data_preparation.feature_construction import FeatureConstructor
from data_preparation.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference on a single essay text file.")
	parser.add_argument(
		"--model-path",
		type=str,
		default="../data/trained_model/trained_model.keras",
		help="Path to a saved Keras model.",
	)
	parser.add_argument(
		"--input-data",
		type=str,
		default="../input_data/ai001.txt",
		help="Path to input essay text file.",
	)
	return parser.parse_args()


def load_text(path: str) -> str:
	with open(path, "r", encoding="utf-8") as handle:
		return handle.read()


def main() -> None:
	args = parse_args()

	if not os.path.exists(args.model_path):
		raise FileNotFoundError(f"Model not found: {args.model_path}")
	if not os.path.exists(args.input_data):
		raise FileNotFoundError(f"Input text not found: {args.input_data}")

	print("Loading model...")
	model = keras.models.load_model(args.model_path)

	print("Reading input text...")
	text = load_text(args.input_data)

	# Build a one-row DataFrame for feature/embedding pipelines.
	df = pd.DataFrame({"text": [text], "label": [0]})

	print("Constructing stylometric features...")
	feature_constructor = FeatureConstructor()
	feature_df = feature_constructor.construct_features(df)
	X_stylo = feature_df.drop(columns=["label"]).values

	print("Creating embeddings...")
	embedder = Tokenizer()
	emb_df = embedder.transform(df)
	X_emb = emb_df.values

	print("Running prediction...")
	prob = model.predict([X_emb, X_stylo], verbose=0)
	prob_value = float(np.squeeze(prob))
	pred_label = int(prob_value >= 0.5)

	print(f"Raw prediction: {pred_label}")
	print(f"Probability (AI=1): {prob_value:.6f}")


if __name__ == "__main__":
	main()


