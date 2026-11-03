import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        out, state = self.lstm(self.embed(x), state)
        return self.fc(self.drop(out)), state
