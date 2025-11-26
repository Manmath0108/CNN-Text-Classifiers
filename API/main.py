from pathlib import Path
import os
import logging
from typing import Optional, Tuple

import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

try:
    from Script.model import TextCNN, load_checkpoint, device as MODEL_DEVICE
except Exception as e:
    raise RuntimeError("Failed to import Script.model. Make sure Script is a package and in sys.path.") from e

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "Models/textcnn_best.pt")
TOKENIZER_NAME  = os.getenv("TOKENIZER_NAME", "distilbert-base-uncased")
MAX_LEN         = int(os.getenv("MAX_LEN", "200"))
EMBED_DIM       = int(os.getenv("EMBED_DIM", "200"))
NUM_FILTERS     = int(os.getenv("NUM_FILTERS", "100"))
KERNEL_SIZES    = tuple(int(x) for x in os.getenv("KERNEL_SIZES", "3,4,5").split(","))
PAD_IDX_OVERRIDE = os.getenv("PAD_IDX", None)
FORCE_CPU       = os.getenv("FORCE_CPU", "0") in ("1", "true", "True")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(title="TextCNN Fake-News Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    probability: float
    prediction: str
    label: int

MODEL: Optional[TextCNN] = None
TOKENIZER = None
DEVICE = None

def select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model_and_tokenizer(
    checkpoint: Path,
    tokenizer_name: str,
    embed_dim: int,
    num_filters: int,
    kernel_sizes: Tuple[int, ...],
    pad_idx: Optional[int],
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    pad_id = pad_idx if pad_idx is not None else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        pad_idx=pad_id,
        dropout=0.5,
    ).to(device)

    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model, tokenizer

def predict_text_single(text: str, model: TextCNN, tokenizer, device: torch.device, max_len: int):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob = torch.sigmoid(logits).cpu().item()

    label = 1 if prob >= 0.5 else 0
    return {
        "text": text,
        "probability": round(prob, 4),
        "prediction": "FAKE" if label == 0 else "REAL",
        "label": label,
    }

@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER, DEVICE
    logging.basicConfig(level=LOG_LEVEL)
    DEVICE = select_device(FORCE_CPU)
    logging.info(f"Selected device: {DEVICE}")

    cp = Path(CHECKPOINT_PATH)
    if not cp.exists():
        logging.error(f"Checkpoint not found at {cp}")
        raise RuntimeError(f"Checkpoint not found: {cp}")

    logging.info("Loading tokenizer and model (this may take a few seconds)...")
    MODEL, TOKENIZER = build_model_and_tokenizer(
        checkpoint=cp,
        tokenizer_name=TOKENIZER_NAME,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        pad_idx=(int(PAD_IDX_OVERRIDE) if PAD_IDX_OVERRIDE is not None else None),
        device=DEVICE,
    )
    logging.info("Model and tokenizer loaded successfully.")

@app.on_event("shutdown")
def shutdown_event():
    global MODEL, TOKENIZER, DEVICE
    MODEL = None
    TOKENIZER = None
    if DEVICE is not None and "cuda" in str(DEVICE):
        torch.cuda.empty_cache()

@app.get("/health")
def health():
    ok = MODEL is not None and TOKENIZER is not None
    return {"ok": ok, "device": str(DEVICE)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not req.text or not isinstance(req.text, str):
        raise HTTPException(status_code=400, detail="Invalid 'text' field (must be non-empty string).")

    try:
        out = predict_text_single(req.text, MODEL, TOKENIZER, DEVICE, MAX_LEN)
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    return out

if __name__ == "__main__":
    uvicorn.run("API.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)