import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================================
# Add project root to Python path
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# =========================================================
# Imports
# =========================================================
from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.masking import generate_mask, apply_mask
from src.contrastive_loss import contrastive_loss


# =========================================================
# Config
# =========================================================
DATA_PATH = "./data/raw/commonvoice_en_au"
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True            # ← change to False for full training
MAX_AUDIO_LEN = 16000   # 1 sec audio


# =========================================================
# Collate Function (handles variable audio length)
# =========================================================
def collate_fn(batch):

    processed = []

    for wave in batch:

        if len(wave) > MAX_AUDIO_LEN:
            wave = wave[:MAX_AUDIO_LEN]
        else:
            pad = MAX_AUDIO_LEN - len(wave)
            wave = torch.nn.functional.pad(wave, (0, pad))

        processed.append(wave)

    return torch.stack(processed)


# =========================================================
# Dataset
# =========================================================
dataset = CommonVoiceSSLDataset(DATA_PATH)

if DEBUG:
    dataset = torch.utils.data.Subset(dataset, range(20))

print("Dataset size:", len(dataset))


# =========================================================
# Loader
# =========================================================
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# =========================================================
# Models
# =========================================================
cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

params = list(cnn.parameters()) + list(transformer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)

print("Device:", DEVICE)


# =========================================================
# Training Loop
# =========================================================
for epoch in range(EPOCHS):

    cnn.train()
    transformer.train()

    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for wave in loop:

        wave = wave.to(DEVICE)

        # Forward pass
        features = cnn(wave)
        encoded = transformer(features)

        # Masking
        B, T, C = encoded.shape
        mask = generate_mask(B, T).to(DEVICE)
        masked = apply_mask(encoded, mask)

        # Loss
        loss = contrastive_loss(masked, encoded)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")


# =========================================================
# Save model
# =========================================================
os.makedirs("checkpoints", exist_ok=True)

torch.save({
    "cnn": cnn.state_dict(),
    "transformer": transformer.state_dict()
}, "checkpoints/ssl_model.pt")

print("\nTraining Complete — Model Saved")