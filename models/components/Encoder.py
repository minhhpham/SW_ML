from torch import nn, Tensor
from models.components import common


class EncoderLayer(nn.Module):
    "Encoder layer made up of attention layers and feed forward layers"

    def __init__(
        self,
        size: int,
        attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float
    ):
        """
        Args:
            size (int): size of attention layer
            attention (nn.Module): attention layer (Q, K, V) -> Y
            feed_forward (nn.Module): feed forward layer (X) => Y
            dropout (float): dropout rate
        """
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.normalize_then_dropout = common.clones(
            module=common.LayerConnections(size, dropout),
            N=2
        )
        self.size = size

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        mask: Tensor
    ) -> Tensor:
        y = self.normalize_then_dropout[0](
            x=x1,
            module=lambda x: self.attention(x, x2, x2, mask)
        )
        return self.normalize_then_dropout[1](y, self.feed_forward)


class Encoder(nn.Module):
    """a stack of N identical EncoderLayers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.encoder_layers = common.clones(layer, N)
        self.standardize = common.Standardize(layer.size)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        mask: Tensor
    ) -> Tensor:
        "Pass the input and mask through each layer. Apply a final standardize"
        for layer in self.encoder_layers:
            x1 = layer(x1, x2, mask)
        return self.standardize(x1)
