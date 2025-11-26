from pathlib import Path
import argparse
import random
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding

from Script.dataset import NewsHFDataset       
from Script.model import TextCNN, load_checkpoint, device as MODEL_DEVICE

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_tokenizer(name: str):
    print(f"Loading tokenizer: {name}")
    return AutoTokenizer.from_pretrained(name, use_fast=True)

def prepare_dataloaders(texts, labels, tokenizer, batch_size, max_len, num_workers=2):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    ds = NewsHFDataset(texts, labels, tokenizer, max_length=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers)
    return loader

def train_epoch(model, loader, optimizer, loss_fn, device, log_every=50):
    model.train()
    running_loss = 0.0
    correct = 0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        n += labels.size(0)

        if step % log_every == 0:
            avg = running_loss / (n if n else 1)
            acc = correct / (n if n else 1)
            print(f"  step {step:04d}  avg_loss={avg:.4f}  acc={acc:.4f}")

    elapsed = time.time() - t0
    return running_loss / (n if n else 1), correct / (n if n else 1), elapsed

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            n += labels.size(0)

    return running_loss / (n if n else 1), correct / (n if n else 1)

def train_and_eval(args):
    device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Using device:", device)

    set_seed(args.seed)

    import pandas as pd
    df = pd.read_csv(args.data_path)

    if "text" not in df.columns:
        if "title" in df.columns:
            df = df.rename(columns={"title": "text"})
        else:
            raise RuntimeError("CSV must contain 'text' or 'title' column.")
    if "label" not in df.columns and "real" in df.columns:
        df = df.rename(columns={"real": "label"})
    if "label" not in df.columns:
        raise RuntimeError("CSV must contain 'label' or 'real' column.")

    df = df[[ "text", "label" ]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if args.max_per_label:
        positives = df[df["label"] == 1].sample(n=min(args.max_per_label, (df["label"]==1).sum()), random_state=args.seed)
        negatives = df[df["label"] == 0].sample(n=min(args.max_per_label, (df["label"]==0).sum()), random_state=args.seed)
        df_small = pd.concat([positives, negatives]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        df_small = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    split = int(len(df_small) * (1 - args.val_frac))
    train_df = df_small.iloc[:split]
    val_df = df_small.iloc[split:]

    print(f"Train size: {len(train_df)}  Val size: {len(val_df)}")

    tokenizer = build_tokenizer(args.tokenizer_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    train_ds = NewsHFDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=args.max_len)
    val_ds = NewsHFDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("Tokenizer:", args.tokenizer_name)
    print("Vocab size:", vocab_size, "Pad id:", pad_id)

    model = TextCNN(vocab_size=vocab_size,
                    embed_dim=args.embed_dim,
                    num_filters=args.num_filters,
                    kernel_sizes=tuple(args.kernel_sizes),
                    pad_idx=pad_id,
                    dropout=args.dropout).to(device)

    if args.load_checkpoint:
        print("Loading checkpoint:", args.load_checkpoint)
        load_checkpoint(args.load_checkpoint, model, map_location=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=False)

    best_val = float("inf")
    no_improve = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "textcnn_best.pt"
    final_path = out_dir / "textcnn_final.pt"

    print("Begin training ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, loss_fn, device, log_every=args.log_every)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        t_elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}  Train Loss = {train_loss:.4f}  Train acc = {train_acc:.4f}  Val Loss = {val_loss:.4f}  Val Acc = {val_acc:.4f}  Time = {t_elapsed:.1f}s")

        scheduler.step(val_loss)

        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved Best Model: {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping No Improvement.")
                break

    if best_path.exists():
        print("Loaded best model for final eval.")
        model.load_state_dict(torch.load(best_path, map_location=device))

    val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
    print(f"Final eval: val_loss = {val_loss:.4f} val_acc = {val_acc:.4f}")

    torch.save(model.state_dict(), final_path)
    print("Saved final model ->", final_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="CSV with text & label columns (title/text and label/real)")
    p.add_argument("--output-dir", type=str, default="Models")
    p.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max-len", dest="max_len", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--embed-dim", dest="embed_dim", type=int, default=200)
    p.add_argument("--num-filters", dest="num_filters", type=int, default=100)
    p.add_argument("--kernel-sizes", nargs="+", type=int, default=[3,4,5])
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-label", dest="max_per_label", type=int, default=800, help="Balanced sampling per class for train (0=use full)")
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="force cpu")
    p.add_argument("--load-checkpoint", type=str, default=None, help="optional checkpoint to initialize model")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    train_and_eval(args)