import torch
import random


def generate_mask(batch_size, time_steps, mask_prob=0.065, mask_length=10):
    """
    Generate time masks for SSL training.
    Returns mask of shape (B, T)
    """
    mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)

    num_masked_spans = int(mask_prob * time_steps / mask_length)

    for b in range(batch_size):
        for _ in range(num_masked_spans):
            start = random.randint(0, time_steps - mask_length)
            mask[b, start:start + mask_length] = True

    return mask


def apply_mask(features, mask):
    """
    Replace masked time steps with zeros
    features: (B, T, C)
    mask: (B, T)
    """
    masked = features.clone()
    masked[mask] = 0
    return masked
