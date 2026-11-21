import math
import torch

BATCH_SIZE = 64
NUM_EPOCHS = 500


def learning_rate_for_epoch(epoch):
    if epoch <= 5:
        return (
            4e-4
            + (epoch - 1)
            * (2e-3 - 4e-4)
            / 4
        )

    if epoch <= 450:
        hi = 2e-3
        lo = 1e-4
        progress = (
            (epoch - 6)
            / (450 - 6)
        )
        return (
            lo
            + (hi - lo)
            * 0.5
            * (
                1
                + math.cos(
                    math.pi * progress
                )
            )
        )

    return 1e-4


def get_optimizer(model):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_for_epoch(1),
        weight_decay=0.0,
    )
    optimizer.vae_epoch = 0
    return optimizer
