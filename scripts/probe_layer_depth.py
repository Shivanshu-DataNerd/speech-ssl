import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")

BATCH_SIZE = 16
EPOCHS = 3
NUM_CLASSES = 3

dataset = CommonVoiceSSLDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

cnn.eval()
transformer.eval()

# Freeze backbone
for p in cnn.parameters():
    p.requires_grad = False

for p in transformer.parameters():
    p.requires_grad = False

# ======================
# Collect one batch
# ======================

waves, labels = next(iter(loader))
waves = waves.to(DEVICE)
labels = labels.to(DEVICE)

with torch.no_grad():
    cnn_out = cnn(waves)
    transformer_layers = transformer(cnn_out, return_all_layers=True)

representations = [cnn_out.mean(dim=1)]

for layer in transformer_layers:
    representations.append(layer.mean(dim=1))

# ======================
# Train probe per layer
# ======================

layer_acc = []

for i, rep in enumerate(representations):

    probe = nn.Linear(rep.shape[1], NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        logits = probe(rep)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.argmax(probe(rep), dim=1)
    acc = (preds == labels).float().mean().item()

    layer_acc.append(acc)

    print(f"Layer {i} Accuracy:", round(acc, 4))

# ======================
# Plot
# ======================

layers = ["CNN"] + [f"T{i}" for i in range(1, len(layer_acc))]

plt.figure(figsize=(6,4))
plt.plot(layers, layer_acc, marker="o")
plt.title("Probe Accuracy vs Layer Depth")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.grid()

save_path = os.path.join(ROOT, "graphs", "probe_depth_accuracy.png")
plt.savefig(save_path, dpi=300)

print("Saved plot:", save_path)