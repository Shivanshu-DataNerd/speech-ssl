import torch
import torch.nn.functional as F


def contrastive_loss(pred, target, temperature=0.1):
    """
    pred: predicted representations (B,T,C)
    target: true representations (B,T,C)

    Computes cosine similarity contrastive loss
    """
    B, T, C = pred.shape

    pred = pred.reshape(-1, C)
    target = target.reshape(-1, C)

    logits = torch.matmul(pred, target.T) / temperature

    labels = torch.arange(pred.shape[0], device=pred.device)

    loss = F.cross_entropy(logits, labels)
    return loss
