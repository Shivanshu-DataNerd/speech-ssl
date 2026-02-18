import torch
from src.cnn_encoder import CNNEncoder

model = CNNEncoder()

dummy = torch.randn(2, 16000)  # batch=2

out = model(dummy)

print("Output shape:", out.shape)
