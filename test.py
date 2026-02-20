import sys
import os
import torch

# --------------------------------------------------
# Add project root to path
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.masking import generate_mask, apply_mask
from src.contrastive_loss import contrastive_loss

# --------------------------------------------------
# Instantiate Models
# --------------------------------------------------
cnn = CNNEncoder()
transformer = TransformerEncoder()

# --------------------------------------------------
# Dummy audio batch
# --------------------------------------------------
dummy = torch.randn(2, 16000)

# --------------------------------------------------
# Forward Pass
# --------------------------------------------------
features = cnn(dummy)
out = transformer(features)

print("Encoder Output Shape:", out.shape)

# --------------------------------------------------
# Masking
# --------------------------------------------------
B, T, C = out.shape

mask = generate_mask(B, T)
masked_features = apply_mask(out, mask)

# --------------------------------------------------
# Contrastive Loss
# --------------------------------------------------
loss = contrastive_loss(masked_features, out)

print("Mask shape:", mask.shape)
print("Loss:", loss.item())
