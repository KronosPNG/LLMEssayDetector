import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path FIRST, then import
sys.path.insert(0, str(Path.cwd() / 'src'))

from data_preparation.feature_construction import FeatureConstructor

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create embeddings CSV from training data.")

	parser.add_argument(
		"--training-data",
		type=str,
		default="../data/training_data/",
		help="Path to training data folder.",
	)
	return parser.parse_args()

args = parse_args()
final_df = pd.DataFrame()

for file in Path(args.training_data).glob("*.csv"):
    print("Processing file:", file)

    print("Loading training data...")
    df = pd.read_csv(file)

    print("Constructing features...")
    fc = FeatureConstructor()
    feature_df = fc.construct_features(df)
    final_df = pd.concat([final_df, feature_df], ignore_index=True)

print("Saving features to CSV...")
final_df.to_csv("../data/train_features.csv", index=False)