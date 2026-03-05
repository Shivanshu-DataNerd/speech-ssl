import torch


def center_gram(gram):
    n = gram.size(0)
    unit = torch.ones(n, n, device=gram.device)
    identity = torch.eye(n, device=gram.device)

    H = identity - unit / n
    return H @ gram @ H


def linear_cka(X, Y):
    """
    Compute linear CKA between two representations.

    X: (N, D1)
    Y: (N, D2)
    """

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    gram_x = X @ X.T
    gram_y = Y @ Y.T

    gram_x = center_gram(gram_x)
    gram_y = center_gram(gram_y)

    hsic = (gram_x * gram_y).sum()

    norm_x = torch.norm(gram_x)
    norm_y = torch.norm(gram_y)

    return hsic / (norm_x * norm_y + 1e-8)