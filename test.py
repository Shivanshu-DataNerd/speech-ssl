import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import CommonVoiceSSLDataset
from src.preprocessing import normalize_waveform, pad_or_trim, random_crop


data_path = os.path.join(PROJECT_ROOT, "data", "raw", "commonvoice_en_au")

dataset = CommonVoiceSSLDataset(data_path)

wave = dataset[0]

print("Original:", wave.shape)

wave = normalize_waveform(wave)
wave = random_crop(wave, 16000)
wave = pad_or_trim(wave, 16000)

print("Processed:", wave.shape)
