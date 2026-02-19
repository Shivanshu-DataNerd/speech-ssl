import sys
import os
import torch

# --------------------------------------------------
# Add project root to Python path (IMPORTANT)
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder


# --------------------------------------------------
# Instantiate Models
# --------------------------------------------------
cnn = CNNEncoder()
transformer = TransformerEncoder()


# --------------------------------------------------
# Dummy waveform batch
# (batch_size=2, samples=16000 → 1 sec audio)
# --------------------------------------------------
dummy_waveform = torch.randn(2, 16000)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------
features = cnn(dummy_waveform)
output = transformer(features)


# --------------------------------------------------
# Print results
# --------------------------------------------------
print("Input shape:", dummy_waveform.shape)
print("CNN output shape:", features.shape)
print("Transformer output shape:", output.shape)
