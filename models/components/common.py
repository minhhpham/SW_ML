import torch
from torch import nn, Tensor
import copy
import math


def clones(module, N):
    "Produce N identical layers of a module object"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Standardize(nn.Module):
    "standardize tensors (x-meam)/std"

    def __init__(self, size: int, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class LayerConnections(nn.Module):
    """
    X -> standardize -> module layer(s) -> dropout
    size: size of output of module layer
    """

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.standardize = Standardize(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, module: nn.Module) -> Tensor:
        return x + self.dropout(module(self.standardize(x)))


class FeedForward(nn.Module):
    """
    x -> Linear -> Relu -> dropout -> linear -> y same dim as x
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x).relu()))


class Embeddings(nn.Module):
    """
    vocabulary index -> vector of size out_dim
    """
    def __init__(self, vocab_size: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.out_dim)


class PositionalEncoding(nn.Module):
    """
    positional encoding with sinusodal of dimension d_model
    PE(pos,2i) = sin(pos/10000 ^ (2i/d_model))
    PE(pos,2i) = cos(pos/10000 ^ (2i/d_model))
    x -> x + PE(x.size(1)) -> dropout
    """

    def __init__(self, d_model: int, dropout: float, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings in log space.
        position_encode = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        position_encode[:, 0::2] = torch.sin(position * div_term)
        position_encode[:, 1::2] = torch.cos(position * div_term)
        position_encode = position_encode.unsqueeze(0)
        self.register_buffer("position_encode", position_encode)

    def forward(self, x: Tensor):
        x = x + self.position_encode[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
