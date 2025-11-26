from pathlib import Path
import argparse
from pprint import pprint
from typing import Dict

import torch
import numpy as np
from transformers import AutoTokenizer

try:
    from Script.model import TextCNN, load_checkpoint, device as MODEL_DEVICE
except Exception as e:
    raise RuntimeError("Failed to import Script.model. Are you running from repo root and Script is a package?") from e


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_and_tokenizer(
    checkpoint: Path,
    tokenizer_name: str,
    embed_dim: int,
    num_filters: int,
    kernel_sizes: tuple,
    pad_idx: int | None,
    device: torch.device,
) -> tuple[TextCNN, AutoTokenizer]:
    """
    Construct model object (same shape as training) and load checkpoint weights.
    Also load HF tokenizer.
    """

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


def predict_text(text: str, model: TextCNN, tokenizer, device: torch.device, max_len: int = 200) -> Dict:
    """
    Tokenize, forward-pass, return probability & label.
    Label mapping: 1 -> FAKE, 0 -> REAL (matches your training labeling).
    """
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


def interactive_loop(model, tokenizer, device, max_len):
    print("Interactive predict mode. Type a sentence and press Enter. Ctrl+C to exit.")
    try:
        while True:
            txt = input("\n> ")
            if not txt.strip():
                print("Empty input, try again.")
                continue
            out = predict_text(txt, model, tokenizer, device, max_len)
            pprint(out)
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


def parse_ks(arg_list):
    if isinstance(arg_list, (list, tuple)):
        return tuple(int(x) for x in arg_list)
    s = str(arg_list)
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    return tuple(int(x) for x in parts if x.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="Models/textcnn_best.pt", help="Path to model checkpoint (.pt)")
    p.add_argument("--tokenizer", type=str, default="distilbert-base-uncased", help="HF tokenizer name")
    p.add_argument("--text", type=str, default=None, help="Single text to predict. If omitted, enters interactive mode.")
    p.add_argument("--max-len", type=int, default=200, help="Max tokens (padding/truncation length)")
    p.add_argument("--embed-dim", type=int, default=200, help="Embed dim used when constructing model (must match training)")
    p.add_argument("--num-filters", type=int, default=100, help="num filters used at training")
    p.add_argument("--kernel-sizes", type=str, default="3,4,5", help="comma separated kernel sizes used at training")
    p.add_argument("--pad-idx", type=int, default=None, help="optional pad idx override (int)")
    p.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA available")
    p.add_argument("--quiet", action="store_true", help="Minimal prints")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    device = get_device(args.force_cpu)
    if not args.quiet:
        print("Using device:", device)
        print("Loading model & tokenizer...")

    ks = parse_ks(args.kernel_sizes)

    model, tokenizer = build_model_and_tokenizer(
        checkpoint=checkpoint,
        tokenizer_name=args.tokenizer,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        kernel_sizes=ks,
        pad_idx=args.pad_idx,
        device=device,
    )

    if args.text:
        out = predict_text(args.text, model, tokenizer, device, max_len=args.max_len)
        pprint(out)
    else:
        interactive_loop(model, tokenizer, device, max_len=args.max_len)


if __name__ == "__main__":
    main()