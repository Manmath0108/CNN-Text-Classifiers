import pandas as pd
from pathlib import Path

def load_fakenews_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    if 'title' not in df.columns or 'real' not in df.columns:
        raise ValueError(f"CSV must contain 'title' and 'real' columns. Found: {df.columns.tolist()}")
    
    df = df.rename(columns={'title': 'text', 'real': 'label'})

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[['text', 'label']].dropna()

    return df