"""Train the LSTM to predict the next note/chord token."""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MusicLSTM
from preprocess import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()

    tokens = load_dataset(args.data)
    vocab = sorted(set(tokens))
    t2i = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([t2i[t] for t in tokens])
    print(f"Vocabulary size: {len(vocab)}")

    # sliding windows: input = seq_len tokens, target = the same window shifted by one
    n = len(ids) - args.seq_len - 1
    X = torch.stack([ids[i:i + args.seq_len] for i in range(n)])
    Y = torch.stack([ids[i + 1:i + args.seq_len + 1] for i in range(n)])
    split = int(0.9 * n)
    train_dl = DataLoader(TensorDataset(X[:split], Y[:split]), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[split:], Y[split:]), batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicLSTM(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out, _ = model(x)
            loss = loss_fn(out.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item()
        model.eval()
        with torch.no_grad():
            va = sum(loss_fn(model(x.to(device))[0].reshape(-1, len(vocab)), y.to(device).reshape(-1)).item()
                     for x, y in val_dl) / max(1, len(val_dl))
        print(f"Epoch {epoch:02d}/{args.epochs}  train loss {tr / len(train_dl):.3f}  val loss {va:.3f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), "model.pt")
            json.dump(vocab, open("vocab.json", "w"))
    print(f"Done. Best val loss {best:.3f}. Saved model.pt and vocab.json")


if __name__ == "__main__":
    main()
