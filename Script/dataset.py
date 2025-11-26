from typing import List
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

MAX_LEN_DEFAULT = 200

class NewsHFDataset(Dataset):
    def __init__ (self, texts: List[str], labels: List[int], tokenizer, max_length: int = MAX_LEN_DEFAULT):
        self.texts = list(texts)

        normalized = []
        for i, lab in enumerate(labels):
            if isinstance(lab, (list, tuple, np.ndarray)):
                if len(lab) == 0:
                    raise ValueError(f"Empty label at index {i}")
                lab0 = lab[0]
            else:
                lab0 = lab
            try:
                lab_int = int(lab0)
            except Exception as e:
                raise TypeError(f"Cannot convert label at index {i} --> {lab0} to int: {e}")
            normalized.append(lab_int)
        self.labels = normalized
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        txt = str(self.texts[idx])

        encoded = self.tokenizer(
            txt,
            truncation = True,
            max_length = self.max_length,
            padding = False,
            return_attention_mask = True,
            return_tensors = None
        )

        encoded["labels"] = int(self.labels[idx])
        return encoded