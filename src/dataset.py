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
        duration_label (Tensor): scalar class label
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        max_duration: float | None = None,
        fixed_length: int | None = 16000
    ):
        """
        Args:
            root_dir: dataset folder path
            sample_rate: resample rate
            max_duration: filter long audio (seconds)
            fixed_length: pad/trim waveform length (samples)
        """

        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio_files")
        self.metadata_path = os.path.join(root_dir, "metadata.csv")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("metadata.csv not found")

        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError("audio_files folder missing")

        self.df = pd.read_csv(self.metadata_path)

        if "path" not in self.df.columns:
            raise ValueError("metadata.csv must contain 'path' column")

        # Optional duration filtering
        if max_duration is not None and "duration_ms" in self.df.columns:
            self.df = self.df[self.df["duration_ms"] / 1000 <= max_duration]

        self.sample_rate = sample_rate
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str):
        full_path = os.path.join(self.audio_dir, path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing audio file: {full_path}")

        signal, _ = librosa.load(full_path, sr=self.sample_rate)
        return signal.astype(np.float32)

    def _fix_length(self, waveform: torch.Tensor):
        """
        Pad or trim waveform to fixed length for batching
        """
        if self.fixed_length is None:
            return waveform

        length = waveform.shape[0]

        if length > self.fixed_length:
            return waveform[:self.fixed_length]

        if length < self.fixed_length:
            pad = self.fixed_length - length
            return torch.nn.functional.pad(waveform, (0, pad))

        return waveform

    def _duration_bucket(self, duration_ms: float):
        """
        Convert duration into class label
        """
        duration_sec = duration_ms / 1000.0

        if duration_sec < 2:
            return 0  # short
        elif duration_sec < 5:
            return 1  # medium
        else:
            return 2  # long

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = row["path"]

        waveform = self._load_audio(audio_path)
        waveform = torch.from_numpy(waveform)

        waveform = self._fix_length(waveform)

        # Duration label
        if "duration_ms" in row:
            label = self._duration_bucket(float(row["duration_ms"]))
        else:
            label = 0  # fallback if column missing

        label = torch.tensor(label, dtype=torch.long)

        return waveform, label