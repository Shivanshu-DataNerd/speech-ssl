import torch
import random


# -------------------------------------------------
# 1. Normalization
# -------------------------------------------------
def normalize_waveform(waveform: torch.Tensor):
    """
    Normalize waveform to zero mean and unit variance.
    """

    mean = waveform.mean()
    std = waveform.std()

    if std < 1e-6:
        return waveform - mean

    return (waveform - mean) / std


# -------------------------------------------------
# 2. Pad or Trim
# -------------------------------------------------
def pad_or_trim(waveform: torch.Tensor, target_length: int):
    """
    Force waveform to fixed length.
    """

    length = waveform.size(0)

    if length > target_length:
        return waveform[:target_length]

    if length < target_length:
        padding = target_length - length
        return torch.nn.functional.pad(waveform, (0, padding))

    return waveform


# -------------------------------------------------
# 3. Random Crop (augmentation)
# -------------------------------------------------
def random_crop(waveform: torch.Tensor, crop_size: int):
    """
    Randomly crop waveform segment.
    """

    length = waveform.size(0)

    if length <= crop_size:
        return waveform

    start = random.randint(0, length - crop_size)
    return waveform[start:start + crop_size]


# -------------------------------------------------
# 4. Collate Function for DataLoader
# -------------------------------------------------
def collate_waveforms(batch, target_length=None):
    """
    Custom collate function for DataLoader.
    """

    processed = []

    for waveform in batch:

        waveform = normalize_waveform(waveform)

        if target_length is not None:
            waveform = pad_or_trim(waveform, target_length)

        processed.append(waveform)

    return torch.stack(processed)
