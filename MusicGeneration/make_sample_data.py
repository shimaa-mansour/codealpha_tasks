"""Creates a small set of sample MIDI files (simple chord progressions with arpeggio
melodies in C major / A minor) so the project runs out of the box.
For better music, put real MIDI files (e.g. classical pieces) into data/ instead."""
import random
import pretty_midi

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
PROGRESSIONS = [[0, 3, 4, 0], [0, 5, 3, 4], [0, 4, 5, 3], [5, 3, 0, 4], [0, 3, 0, 4]]
PATTERNS = [[0, 1, 2, 1, 0, 1, 2, 1], [0, 2, 1, 2, 0, 2, 1, 2], [0, 1, 2, 3, 2, 1, 0, 1], [2, 1, 0, 1, 2, 3, 2, 1]]


def make_piece(seed: int, path: str):
    rnd = random.Random(seed)
    scale = MAJOR if seed % 2 == 0 else MINOR
    root = 60 if scale is MAJOR else 57
    prog = rnd.choice(PROGRESSIONS)
    pattern = rnd.choice(PATTERNS)
    beat = 0.5  # seconds (120 bpm)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for bar in range(16):
        d = prog[bar % 4]
        tones = [scale[(d + k) % 7] + 12 * ((d + k) // 7) for k in (0, 2, 4, 7)]  # triad + octave
        for k in (0, 1, 2):  # chord held for the bar
            inst.notes.append(pretty_midi.Note(70, root - 12 + tones[k], t, t + 4 * beat))
        for step, idx in enumerate(pattern):
            if rnd.random() < 0.08:          # small random variation
                idx = rnd.choice([0, 1, 2, 3])
            pitch = root + tones[idx]
            inst.notes.append(pretty_midi.Note(90, pitch, t + step * 0.5 * beat, t + (step + 1) * 0.5 * beat))
        t += 4 * beat
    pm.instruments.append(inst)
    pm.write(path)


if __name__ == "__main__":
    for i in range(60):
        make_piece(i, f"data/sample_{i:02d}.mid")
    print("Created 60 sample MIDI files in data/")
