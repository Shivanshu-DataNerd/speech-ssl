import torch.nn as nn
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.projector import ProjectionHead


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.transformer = TransformerEncoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        x = self.cnn(x)
        x = self.transformer(x)
        x = self.projector(x)
        return x