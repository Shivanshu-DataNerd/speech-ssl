import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class CommonVoiceSSLDataset(Dataset):
    """
    Self-Supervised Learning Dataset for Speech Models

    Loads raw waveform audio from CommonVoice dataset.

    Returns:
        waveform (Tensor): shape (T,)
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        max_duration: float | None = None
    ):
        """
        Args:
            root_dir: path to dataset folder
            sample_rate: target sample rate
            max_duration: optional filter to remove long files
        """

        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio_files")
        self.metadata_path = os.path.join(root_dir, "metadata.csv")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("metadata.csv not found")

        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError("audio_files folder missing")

        # Load metadata
        self.df = pd.read_csv(self.metadata_path)

        # Validate required column
        if "path" not in self.df.columns:
            raise ValueError("metadata.csv must contain 'path' column")

        # Optional filtering by duration
        if max_duration is not None and "duration_ms" in self.df.columns:
            self.df = self.df[self.df["duration_ms"] / 1000 <= max_duration]

        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str):
        """
        Load audio file safely using librosa.
        Returns float32 waveform.
        """

        full_path = os.path.join(self.audio_dir, path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing audio file: {full_path}")

        signal, _ = librosa.load(full_path, sr=self.sample_rate)
        return signal.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = row["path"]

        waveform = self._load_audio(audio_path)

        # Convert to tensor
        waveform = torch.from_numpy(waveform)

        return waveform
