import torch
import torch.nn as nn


class DurationProbe(nn.Module):
    """
    Linear probing head for duration classification.
    """

    def __init__(self, encoder, embedding_dim, num_classes=3):
        super().__init__()

        # Freeze encoder
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Linear classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        x: waveform batch (B, T)
        """

        # Extract features from frozen encoder
        features = self.encoder(x)  # (B, time, C)

        # Mean pooling across time dimension
        features = features.mean(dim=1)  # (B, C)

        logits = self.classifier(features)

        return logits


        