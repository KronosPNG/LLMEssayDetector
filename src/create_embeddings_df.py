import pandas as pd
import sys
from pathlib import Path

# Add src directory to path FIRST, then import
sys.path.insert(0, str(Path.cwd() / 'src'))

from data_preparation.tokenizer import Tokenizer

print("Loading training data...")
df = pd.read_csv("../data/train_drcat_01.csv")
df = df.drop(columns=["source", "fold"])

print("Creating embeddings...")
embedder = Tokenizer()
emb_df = embedder.transform_and_save(df, path="../data/train_embeddings.csv", keep_cols=["label"])
