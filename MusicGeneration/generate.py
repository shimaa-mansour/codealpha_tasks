"""Generate a new piece with the trained LSTM and save it as a MIDI file."""
import argparse
import json
import random

import torch

from model import MusicLSTM
from preprocess import load_dataset, tokens_to_midi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=120, help="number of notes/chords to generate")
    p.add_argument("--temperature", type=float, default=0.9, help="lower = safer, higher = more random")
    p.add_argument("--tempo", type=float, default=120)
    p.add_argument("--seed-len", type=int, default=16)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="generated.mid")
    args = p.parse_args()

    vocab = json.load(open("vocab.json"))
    t2i = {t: i for i, t in enumerate(vocab)}
    model = MusicLSTM(len(vocab))
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    # start from a random snippet of the training data
    source = [t for t in load_dataset(args.data) if t in t2i]
    start = random.randint(0, len(source) - args.seed_len - 1)
    generated = source[start:start + args.seed_len]

    x = torch.tensor([[t2i[t] for t in generated]])
    with torch.no_grad():
        out, state = model(x)
        for _ in range(args.length):
            probs = torch.softmax(out[0, -1] / args.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(vocab[nxt])
            out, state = model(torch.tensor([[nxt]]), state)

    tokens_to_midi(generated, args.out, args.tempo)
    print(f"Saved {args.out} ({len(generated)} notes/chords). Open it with any MIDI player.")


if __name__ == "__main__":
    main()
