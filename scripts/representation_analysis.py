import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")

NUM_SAMPLES = 512

dataset = CommonVoiceSSLDataset(DATA_PATH)

cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

cnn.eval()
transformer.eval()

# ======================
# Collect embeddings
# ======================

waves = []

for i in range(NUM_SAMPLES):
    wave, _ = dataset[i]
    waves.append(wave)

waves = torch.stack(waves).to(DEVICE)

with torch.no_grad():

    x = cnn(waves)
    x = transformer(x)

    embeddings = x.mean(dim=1)

embeddings = embeddings.cpu().numpy()

print("Embedding shape:", embeddings.shape)

# ======================
# Eigenvalue Spectrum
# ======================

cov = np.cov(embeddings, rowvar=False)

eigvals = np.linalg.eigvalsh(cov)
eigvals = np.sort(eigvals)[::-1]

plt.figure(figsize=(6,4))
plt.plot(eigvals)
plt.title("Eigenvalue Spectrum of Representations")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")

save_path = os.path.join(ROOT, "graphs", "eigenvalue_spectrum.png")
plt.savefig(save_path, dpi=300)

print("Saved eigenvalue spectrum")

# ======================
# Feature Anisotropy
# ======================

norms = np.linalg.norm(embeddings, axis=1)

anisotropy = np.std(norms) / np.mean(norms)

print("\nFeature Anisotropy:", round(anisotropy, 4))

# ======================
# Collapse Detection
# ======================

variance = np.var(embeddings, axis=0)
collapse_score = np.mean(variance)

print("\nRepresentation Variance:", round(collapse_score, 6))

if collapse_score < 1e-5:
    print("Representation Collapse Detected")
else:
    print(" Representations are healthy")