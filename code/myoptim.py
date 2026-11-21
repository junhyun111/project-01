
import math
import torch

BATCH_SIZE = 32
NUM_EPOCHS = 500

def get_optimizer(model):
    lr = 5e-4
    eta_min = 1e-5
    decay_end = 300

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    def lr_lambda(epoch):
        t = min(epoch, decay_end)
        return (
            eta_min / lr
            + (1 - eta_min / lr)
            * (1 + math.cos(math.pi * t / decay_end)) / 2
        )

    optimizer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    return optimizer
