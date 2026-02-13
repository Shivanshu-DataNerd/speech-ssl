import os
import torch
import torchaudio
from torch.utils.data import Dataset

SUPPORTED_FORMATS = (".wav", ".flac", ".mp3", ".ogg")


class RawAudioDataset(Dataset):
    """
    Loads raw waveform audio files for SSL / ASR training.
    """

    def __init__(self, root_dir, sample_rate=16000):
        """
        Args:
            root_dir: path to dataset root
                      e.g. data/raw/commonvoice_en_au
        """

        self.audio_dir = os.path.join(root_dir, "audio_files")
        self.sample_rate = sample_rate

        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}"
            )

        self.files = self._collect_files()

        if len(self.files) == 0:
            raise RuntimeError("No audio files found.")

    def _collect_files(self):
        paths = []

        for f in os.listdir(self.audio_dir):
            if f.lower().endswith(SUPPORTED_FORMATS):
                paths.append(os.path.join(self.audio_dir, f))

        return sorted(paths)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        # stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # normalize
        waveform = waveform / waveform.abs().max()

        return waveform
