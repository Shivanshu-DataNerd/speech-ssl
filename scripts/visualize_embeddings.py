import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.manifold import TSNE
import umap

# add root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder


# ======================
# Config
# ======================

DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 500   # keep small for speed
METHOD = "tsne"     # options: tsne | umap


# ======================
# Load Dataset
# ======================

dataset = CommonVoiceSSLDataset(DATA_PATH)

print("Dataset size:", len(dataset))


# ======================
# Load Encoder
# ======================

cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

cnn.eval()
transformer.eval()


# ======================
# Extract Embeddings
# ======================

embeddings = []
labels = []

print("Extracting embeddings...")

for i in range(NUM_SAMPLES):
    wave, label = dataset[i]

    wave = wave.unsqueeze(0).to(DEVICE)  # (1, T)

    with torch.no_grad():
        x = cnn(wave)
        x = transformer(x)

        # mean pooling
        x = x.mean(dim=1)

    embeddings.append(x.squeeze().cpu().numpy())
    labels.append(label.item())

embeddings = np.array(embeddings)
labels = np.array(labels)

print("Embedding shape:", embeddings.shape)


# ======================
# Dimensionality Reduction
# ======================

print("Running", METHOD)

if METHOD == "tsne":
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
elif METHOD == "umap":
    reducer = umap.UMAP(n_components=2, random_state=42)
else:
    raise ValueError("Invalid method")

proj = reducer.fit_transform(embeddings)


# ======================
# Plot
# ======================

plt.figure(figsize=(8,6))
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=labels, palette="tab10", s=40)

plt.title(f"Embedding Visualization ({METHOD.upper()})")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(title="Class")
plt.tight_layout()

save_path = os.path.join(ROOT, "graphs", f"embedding_{METHOD}.png")
plt.savefig(save_path, dpi=300)

print("Saved plot →", save_path)