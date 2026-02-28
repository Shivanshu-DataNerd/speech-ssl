import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# add root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.probes import DurationProbe

# -------------------------
# Config
# -------------------------
DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")
CHECKPOINT = os.path.join(ROOT, "checkpoints/duration_probe.pt")

BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Dataset
# -------------------------
dataset = CommonVoiceSSLDataset(DATA_PATH, max_duration=10)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

print("Dataset size:", len(dataset))

# -------------------------
# Encoder
# -------------------------
cnn = CNNEncoder()
transformer = TransformerEncoder()
encoder = nn.Sequential(cnn, transformer).to(device)

# -------------------------
# Probe
# -------------------------
probe = DurationProbe(
    encoder=encoder,
    embedding_dim=512,
    num_classes=3
).to(device)

probe.load_state_dict(torch.load(CHECKPOINT, map_location=device))
probe.eval()

print("Loaded trained probe.")

# -------------------------
# Evaluation
# -------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for wave, label in loader:

        wave = wave.to(device)
        logits = probe(wave)

        preds = torch.argmax(logits, dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(label.tolist())

# -------------------------
# Metrics
# -------------------------
acc = accuracy_score(all_labels, all_preds)

print("\nAccuracy:", round(acc, 4))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))