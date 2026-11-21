"""Optimization settings: batch size, number of epochs, optimizer."""

import torch

# Used by the notebook.
BATCH_SIZE = 32     # number of images in a mini-batch
NUM_EPOCHS = 200    # number of passes over the training data


def get_optimizer(model):
    """Create the optimizer that updates the parameters of `model`.

    The choice of the optimizer and of its settings (learning rate, ...) is
    yours.

    Args:
        model: the VAE defined in mymodel.py.

    Returns:
        a torch.optim.Optimizer.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    optimizer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )
    return optimizer
