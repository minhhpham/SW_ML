import torch
from torch import nn, Tensor
from models.components.Attention import MultiHeadedAttention
from models.components.common import (
    FeedForward, PositionalEncoding, Embeddings
)
from models.components.Encoder import Encoder, EncoderLayer


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        stack_size: int,
        d_model=512,
        d_feed_fwd=2048,
        n_attn_heads=8,
        dropout=0.1
    ) -> None:
        """
        vocab_size: number of words in vocab
        stack_size: number of stacked encoding layers
        d_model: output size of attention, feed forward, and position encoding
        d_feed_fwd: hidden layer size in feed forward
        n_attn_heads: number of attention heads
        dropout: dropout rate
        """
        super().__init__()
        self.embed = Embeddings(vocab_size=vocab_size, out_dim=d_model)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.attention = MultiHeadedAttention(
            n_heads=n_attn_heads,
            size=d_model,
            dropout=dropout
        )
        self.feed_fwd = FeedForward(
            input_dim=d_model,
            hidden_dim=d_feed_fwd,
            dropout=dropout
        )
        self.encoder = Encoder(
            EncoderLayer(
                size=d_model,
                attention=self.attention,
                feed_forward=self.feed_fwd,
                dropout=dropout
            ),
            N=stack_size
        )
        # Important: Initialize parameters with Glorot / fan_avg.
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # linear output
        self.linear_output = nn.Linear(d_model*24, 1)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        mask: Tensor
    ) -> Tensor:
        x = self.encoder(
            x1=self.position(self.embed(x1)),
            x2=self.position(self.embed(x2)),
            mask=mask
        )
        return self.linear_output(torch.flatten(x, start_dim=1)).relu()
