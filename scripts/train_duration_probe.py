import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# add root
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

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------
# Dataset
# -------------------------
dataset = CommonVoiceSSLDataset(DATA_PATH, max_duration=10)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))

# -------------------------
# Encoder (frozen backbone)
# -------------------------
cnn = CNNEncoder()
transformer = TransformerEncoder()

encoder = nn.Sequential(cnn, transformer)
encoder.to(device)

# -------------------------
# Probe
# -------------------------
probe = DurationProbe(
    encoder=encoder,
    embedding_dim=512,
    num_classes=3
).to(device)

# -------------------------
# Optimizer
# -------------------------
optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):

    probe.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for wave, label in loop:

        wave = wave.to(device)
        label = label.to(device)

        logits = probe(wave)

        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Avg Loss:", round(avg_loss, 4))


# -------------------------
# Save Probe
# -------------------------
os.makedirs("checkpoints", exist_ok=True)

torch.save(probe.state_dict(), "checkpoints/duration_probe.pt")

print("\nTraining complete — probe saved.")