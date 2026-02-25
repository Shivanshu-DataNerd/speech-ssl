import os 
import sys
from src.dataset import CommonVoiceSSLDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "commonvoice_en_au")
ds = CommonVoiceSSLDataset(DATA_PATH)
print("Dataset size:", len(ds))



wave, label = ds[0]

print(wave.shape)
print(label)