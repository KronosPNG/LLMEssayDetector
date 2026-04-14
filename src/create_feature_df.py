import pandas as pd
import sys
from pathlib import Path

# Add src directory to path FIRST, then import
sys.path.insert(0, str(Path.cwd() / 'src'))

from data_preparation.feature_construction import FeatureConstructor

print("Loading training data...")
df = pd.read_csv("../data/train_drcat_01.csv")
df = df.drop(columns=["source", "fold"])

print("Constructing features...")
fc = FeatureConstructor()
feature_df = fc.construct_features(df)
feature_df.to_csv("../data/train_features.csv", index=False)