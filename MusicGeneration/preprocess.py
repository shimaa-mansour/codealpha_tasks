"""MIDI -> sequence of tokens. A token is one note or chord: '60.64.67_4'
(pitches joined by '.', then '_' and the duration in 16th-note steps)."""
import glob
import os

import pretty_midi

MAX_DUR = 16          # longest duration token (in 16th notes)
REST = "R"


def midi_to_tokens(path: str) -> list:
    pm = pretty_midi.PrettyMIDI(path)
    tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) else 120.0
    step = 60.0 / tempo / 4                      # seconds per 16th note
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))

    # group notes that start together into chords
    groups, i = [], 0
    while i < len(notes):
        j = i
        while j < len(notes) and abs(notes[j].start - notes[i].start) < step / 2:
            j += 1
        chunk = notes[i:j]
        groups.append((chunk[0].start, sorted({n.pitch for n in chunk}), min(n.end for n in chunk) - chunk[0].start))
        i = j

    tokens, prev_end = [], None
    for start, pitches, dur in groups:
        if prev_end is not None:
            gap = round((start - prev_end) / step)
            if gap >= 1:
                tokens.append(f"{REST}_{min(gap, MAX_DUR)}")
        d = max(1, min(MAX_DUR, round(dur / step)))
        tokens.append(".".join(map(str, pitches)) + f"_{d}")
        prev_end = start + d * step
    return tokens


def load_dataset(folder: str = "data") -> list:
    files = glob.glob(os.path.join(folder, "**", "*.mid*"), recursive=True)
    if not files:
        raise SystemExit(f"No MIDI files found in '{folder}/'. Run make_sample_data.py or add your own .mid files.")
    all_tokens = []
    for f in files:
        try:
            all_tokens.extend(midi_to_tokens(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    print(f"Loaded {len(files)} files -> {len(all_tokens)} tokens")
    return all_tokens


def tokens_to_midi(tokens: list, path: str, tempo: float = 120.0):
    step = 60.0 / tempo / 4
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)  # acoustic grand piano
    t = 0.0
    for tok in tokens:
        body, dur = tok.rsplit("_", 1)
        length = int(dur) * step
        if body != REST:
            for p in body.split("."):
                inst.notes.append(pretty_midi.Note(velocity=85, pitch=int(p), start=t, end=t + length))
        t += length
    pm.instruments.append(inst)
    pm.write(path)
