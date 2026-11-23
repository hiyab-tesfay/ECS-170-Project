"""Own sequence classifier and causal language model with interchangeable cells."""

import torch
from torch import nn


def recurrent(cell, input_size, hidden_size, layers, dropout, bidirectional=False):
    return {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell](
        input_size, hidden_size, num_layers=layers, batch_first=True,
        dropout=dropout if layers > 1 else 0, bidirectional=bidirectional)


def initialize(module):
    """Xavier input/output weights, gate-wise orthogonal recurrent weights."""
    for name, p in module.named_parameters():
        if 'embedding' in name:
            continue
        if 'weight_hh' in name:
            for gate in p.chunk(p.shape[0] // p.shape[1], dim=0):
                nn.init.orthogonal_(gate)
        elif p.ndim >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=128, cell='gru',
                 layers=1, dropout=0.4, bidirectional=True, pretrained=None,
                 freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad_(not freeze_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.encoder = recurrent(cell, embedding_dim, hidden_size, layers, dropout)
        self.reverse_encoder = (recurrent(cell, embedding_dim, hidden_size, layers, dropout)
                                if bidirectional else None)
        width = hidden_size * (2 if bidirectional else 1)
        self.output = nn.Linear(width * 2, 2)
        initialize(self)

    def forward(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        embeddings = self.dropout(self.embedding(tokens))
        states, _ = self.encoder(embeddings)
        if self.reverse_encoder is not None:
            # Reverse only actual words. Both recurrent passes then see padding
            # AFTER the words, and valid outputs cannot depend on padded steps.
            # This retains optimized dense kernels rather than packed kernels.
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            lengths = mask.squeeze(-1).sum(1, keepdim=True)
            reverse = (lengths - 1 - positions).clamp_min(0).unsqueeze(-1)
            backwards = embeddings.gather(1, reverse.expand_as(embeddings))
            backwards, _ = self.reverse_encoder(backwards)
            backwards = backwards.gather(1, reverse.expand(-1, -1, backwards.shape[-1]))
            states = torch.cat([states, backwards], dim=-1)
        mean = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        maximum = states.masked_fill(~mask, -1e4).max(1).values
        return self.output(self.dropout(torch.cat([mean, maximum], dim=1)))


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=256, cell='lstm',
                 layers=1, dropout=0.2, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained))
        self.embedding.weight.data[0].zero_()
        self.dropout = nn.Dropout(dropout)
        # Bidirectionality would leak future target words during generation.
        self.encoder = recurrent(cell, embedding_dim, hidden_size, layers, dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
        initialize(self)

    def forward(self, tokens, state=None):
        hidden, state = self.encoder(self.dropout(self.embedding(tokens)), state)
        return self.output(self.dropout(hidden)), state
