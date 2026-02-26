import torch

from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder

# Create encoder stack
cnn = CNNEncoder()
transformer = TransformerEncoder()

# Combine into one encoder module
class FullEncoder(torch.nn.Module):
    def __init__(self, cnn, transformer):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer

    def forward(self, x):
        x = self.cnn(x)
        x = self.transformer(x)
        return x


encoder = FullEncoder(cnn, transformer)

# Test input
dummy = torch.randn(2, 16000)

# Forward pass
features = encoder(dummy)

print("Encoder output shape:", features.shape)