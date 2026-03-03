import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.cka import linear_cka

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")

NUM_SAMPLES = 128

dataset = CommonVoiceSSLDataset(DATA_PATH)

cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

cnn.eval()
transformer.eval()

# ======================
# Collect batch
# ======================

waves = []
for i in range(NUM_SAMPLES):
    wave, _ = dataset[i]
    waves.append(wave)

waves = torch.stack(waves).to(DEVICE)

with torch.no_grad():
    cnn_out = cnn(waves)  # (B, T, C)

    transformer_layers = transformer(cnn_out, return_all_layers=True)

# Mean pool all representations
representations = []

# CNN pooled
representations.append(cnn_out.mean(dim=1))

# Transformer layers pooled
for layer_out in transformer_layers:
    representations.append(layer_out.mean(dim=1))

num_layers = len(representations)

cka_matrix = torch.zeros(num_layers, num_layers)

for i in range(num_layers):
    for j in range(num_layers):
        cka_matrix[i, j] = linear_cka(
            representations[i],
            representations[j]
        )

cka_matrix = cka_matrix.cpu().numpy()

# ======================
# Plot Heatmap
# ======================

labels = ["CNN"] + [f"T{i+1}" for i in range(num_layers-1)]

plt.figure(figsize=(8,6))
sns.heatmap(
    cka_matrix,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    cmap="viridis",
    fmt=".2f"
)

plt.title("Layer-wise CKA Similarity")
plt.tight_layout()

save_path = os.path.join(ROOT, "graphs", "layerwise_cka.png")
plt.savefig(save_path, dpi=300)

print("Saved heatmap to:", save_path)