from torch import nn, Tensor
import torch
from typing import Tuple
from models.components import common
import math


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
    dropout: nn.Module = None
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot Product Attention
    softmax(Q transpose(T))/sqrt(dim) V
    Return result weighted value and attention weights
    """
    size = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(size)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = scores.softmax(dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadedAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            size: int,
            dropout=0.1):
        """Multi head attention

        Args:
            n_heads (int): number of heads
            size (int): dimension of query, key value
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        assert size % n_heads == 0, "input size must be divisible by n_heads"
        # We assume d_v always equals d_k
        self.d_k = size // n_heads
        self.n_heads = n_heads
        # 3 linear layers, each for Q, K, V + 1 linear layer at the end
        self.linears = common.clones(nn.Linear(size, size), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None
    ):
        # Same mask applied to all n_heads heads.
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from size => n_heads x d_k
        query, key, value = [
            linear(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2) # noqa
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.n_heads * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
