from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

class Tokenizer:
    def __init__(
            self, 
            model_name="allenai/scibert_scivocab_uncased",
            max_length=512,
            batch_size=16,
            text_col="text",
            truncation="first_last",
            pooling_strategy="mean",
            device=None   
        ):

        self.max_len = max_length
        self.batch_size = batch_size
        self.text_col = text_col
        self.truncation = truncation
        self.pooling = pooling_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
    
    def _tokenize(self, text):
        if self.truncation == "first_last":
            if self.truncation == "first_last":
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                half   = (self.max_len - 2) // 2

                if len(tokens) > self.max_len - 2:
                    tokens = tokens[:half] + tokens[-half:]

                tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
                padding_len = self.max_len - len(tokens)
                attention_mask = [1] * len(tokens) + [0] * padding_len
                input_ids = tokens + [self.tokenizer.pad_token_id] * padding_len
                
                return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        else:  # "head" — just take the first 512 tokens
            return self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )
        
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Reduces [batch, seq_len, hidden] → [batch, hidden]."""
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
        elif self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
    
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of raw strings → numpy array [batch, hidden]."""
        batch_enc = [self._tokenize(t) for t in texts]
        input_ids = torch.tensor([e["input_ids"]      for e in batch_enc]).to(self.device)
        attn_mask = torch.tensor([e["attention_mask"] for e in batch_enc]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)

        pooled = self._pool(outputs.last_hidden_state, attn_mask)
        return pooled.cpu().numpy()
    
    def transform(self, df: pd.DataFrame, keep_cols: list[str] = None) -> pd.DataFrame:
        """
        Main method. Accepts a DataFrame, returns a new DataFrame with
        embedding columns (emb_0 … emb_N) plus any columns in keep_cols.

        Usage:
            embedder = EssayEmbedder()
            emb_df   = embedder.transform(df, keep_cols=["label"])
        """
        texts  = df[self.text_col].fillna("").tolist()
        chunks = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]

        all_embeddings = []
        for batch in tqdm(chunks, desc="Embedding batches"):
            all_embeddings.append(self._embed_batch(batch))

        embeddings = np.vstack(all_embeddings)  # [N, hidden]
        col_names  = [f"emb_{i}" for i in range(embeddings.shape[1])]
        result     = pd.DataFrame(embeddings, columns=col_names, index=df.index)

        if keep_cols:
            result[keep_cols] = df[keep_cols].values

        return result

    def transform_and_save(self, df: pd.DataFrame, path: str, keep_cols: list[str] = None):
        """Convenience wrapper — embeds and saves to parquet in one call."""
        emb_df = self.transform(df, keep_cols=keep_cols)
        emb_df.to_csv(path, index=False)
        print(f"Saved {emb_df.shape} → {path}")

        return emb_df