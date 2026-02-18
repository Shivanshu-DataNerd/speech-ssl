import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Raw waveform → latent representation encoder
    """

    def __init__(self, in_channels=1, hidden_dim=512):
        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv1d(in_channels, 64, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        x shape: (batch, time)
        returns: (batch, feature_dim, frames)
        """

        x = x.unsqueeze(1)  # add channel dim
        x = self.conv_layers(x)
        return x
