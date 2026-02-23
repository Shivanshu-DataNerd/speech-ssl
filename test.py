import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.projector import ProjectionHead
from src.masking import generate_mask, apply_mask
from src.contrastive_loss import contrastive_loss

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------------------------------------------------
# CONFIG (small debug mode)
# --------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "commonvoice_en_au")

BATCH_SIZE = 4
EPOCHS = 5
DEBUG_STEPS = 5      # only run few batches for testing

# --------------------------------------------------
# Dataset + Loader
# --------------------------------------------------
dataset = CommonVoiceSSLDataset(DATA_PATH)
print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# --------------------------------------------------
# Models
# --------------------------------------------------
cnn = CNNEncoder().to(device)
transformer = TransformerEncoder().to(device)
projector = ProjectionHead(input_dim=256).to(device)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
optimizer = torch.optim.Adam(
    list(cnn.parameters())
    + list(transformer.parameters())
    + list(projector.parameters()),
    lr=1e-4
)

# --------------------------------------------------
# Training
# --------------------------------------------------
loss_history = []

for epoch in range(EPOCHS):

    running_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, wave in enumerate(loop):

        if step >= DEBUG_STEPS:
            break

        wave = wave.to(device)

        # ----------------------------
        # Forward Pass
        # ----------------------------
        features = cnn(wave)
        encoded = transformer(features)
        projected = projector(encoded)

        B, T, C = projected.shape

        # ----------------------------
        # Masking
        # ----------------------------
        mask = generate_mask(B, T).to(device)
        masked_features = apply_mask(projected, mask)

        # ----------------------------
        # Loss
        # ----------------------------
        loss = contrastive_loss(masked_features, projected)

        # ----------------------------
        # Backprop
        # ----------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ----------------------------
        # Metrics
        # ----------------------------
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / DEBUG_STEPS
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

# --------------------------------------------------
# Save model
# --------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)

torch.save({
    "cnn": cnn.state_dict(),
    "transformer": transformer.state_dict(),
    "projector": projector.state_dict()
}, "checkpoints/ssl_model.pt")

print("\nTraining Complete — Model Saved")