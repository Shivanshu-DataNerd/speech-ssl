import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Temporal Context Encoder
    Converts CNN features into contextual speech representations
    """

    def __init__(
        self,
        input_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()

        self.pos_embedding = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x shape: (batch, feature_dim, frames)
        returns: (batch, frames, feature_dim)
        """

        x = x.transpose(1, 2)  # (batch, frames, feature_dim)

        x = self.pos_embedding(x)
        x = self.transformer(x)

        return x


# -------------------------------------------------
# Positional Encoding
# -------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, return_all_layers=False):
        layer_outputs = []

        for layer in self.layers:
            x = layer(x)
            if return_all_layers:
                layer_outputs.append(x)

        if return_all_layers:
            return layer_outputs  # list of (B, T, C)

        return x
