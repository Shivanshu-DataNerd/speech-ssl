import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.cnn_encoder import CNNEncoder
from src.transformer_encoder import TransformerEncoder
from src.cka import linear_cka


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(ROOT, "data/raw/commonvoice_en_au")

dataset = CommonVoiceSSLDataset(DATA_PATH)

cnn = CNNEncoder().to(DEVICE)
transformer = TransformerEncoder().to(DEVICE)

cnn.eval()
transformer.eval()

NUM_SAMPLES = 128

waves = []

for i in range(NUM_SAMPLES):
    wave, _ = dataset[i]
    waves.append(wave)

waves = torch.stack(waves).to(DEVICE)

with torch.no_grad():
    cnn_out = cnn(waves)              # (B, T, C)
    transformer_out = transformer(cnn_out)

    # mean pool
    cnn_feat = cnn_out.mean(dim=1)
    transformer_feat = transformer_out.mean(dim=1)

cka_score = linear_cka(cnn_feat, transformer_feat)

print("CKA Similarity Score:", round(cka_score.item(), 4))
print("CKA Similarity Score:", round(cka_score.item(), 4))