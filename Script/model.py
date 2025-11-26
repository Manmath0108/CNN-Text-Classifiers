from typing import Sequence, Optional, Tuple, Dict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int = 200, num_filters:int = 128, kernel_sizes:Sequence[int] = (3, 4, 5), pad_idx:int = 0, dropout:float = 0.5, freeze_embeddings:bool = False, pretrained_embeddings:Optional[torch.Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, torch.Tensor):
                emb_tensor = pretrained_embeddings
            else:
                emb_tensor = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            assert emb_tensor.shape == (vocab_size, embed_dim), f"Pretrained embeddings must be shape ({vocab_size}, {embed_dim})"
            self.embedding.weight.data.copy_(emb_tensor)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embed_dim))
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

        nn.init.xavier_uniform_(self.embedding.weight)
        for c in self.convs:
            nn.init.kaiming_uniform_(c.weight, nonlinearity="relu")
            if c.bias is not None:
                nn.init.zeros_(c.bias)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            emb = emb * mask
        emb = emb.unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            c = conv(emb)
            c = F.relu(c)
            c = c.squeeze(3)
            c = c.max(dim=2).values
            conv_outputs.append(c)
        cat = torch.cat(conv_outputs, dim=1)
        x = self.dropout(cat)
        logits = self.classifier(x).squeeze(-1)
        return logits
    
    def get_embedding_weights(self) -> torch.Tensor:
        return self.embedding.weight.data.detach().cpu()
    
def load_checkpoint(path: str | Path, model: nn.Module, map_location: Optional[str | torch.device] = None) -> Dict:
    p = Path(path)
    assert p.exists(), f"Checkpoint not Found: {p}"
    state = torch.load(str(p), map_location=map_location or device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and all(k.startswith("embedding.") or k.startswith("convs.") or k.startswith("classifier.") or "." in k for k in state.keys()):
        state_dict = state
    else:
        state_dict = state
    
    model.load_state_dict(state_dict)
    return state_dict

__all__ = ["TextCNNNotebook", "TextCNN", "load_checkpoint", "device"]