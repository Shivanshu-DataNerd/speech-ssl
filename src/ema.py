import copy
import torch


class EMA:
    """
    Exponential Moving Average model copy.
    Used as stable teacher network.
    """

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

        for p in self.model.parameters():
            p.requires_grad = False

    def update(self, student_model):
        with torch.no_grad():
            for ema_p, student_p in zip(
                self.model.parameters(),
                student_model.parameters()
            ):
                ema_p.data.mul_(self.decay).add_(
                    student_p.data, alpha=1 - self.decay
                )