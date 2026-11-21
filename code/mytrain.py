"""Training of the VAE on original images with BCE + KL loss."""

import torch.nn.functional as F


def loss_function(x, x_hat, mu, logvar):
    """VAE loss (negative ELBO) of a batch, averaged over its images.

        loss  = recon + kl

        recon : reconstruction loss, the binary cross-entropy between x_hat
                and x, summed over the pixels of an image
        kl    : KL divergence KL( q(z|x) || p(z) ) with the prior
                p(z) = N(0, I), summed over the latent dimensions

    Both terms are averaged over the B images of the batch.

    Args:
        x:      images,                  shape (B, 1, 32, 32), values in [0, 1].
        x_hat:  decoded images,          shape (B, 1, 32, 32), values in [0, 1].
        mu:     mean of q(z|x),          shape (B, latent_dim).
        logvar: log-variance of q(z|x),  shape (B, latent_dim).

    Returns:
        (loss, recon, kl): three scalar tensors.
    """
    batch_size = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / batch_size
    return recon + kl, recon, kl


def train_one_epoch(model, optimizer, dataloader, device):
    """Train `model` for one epoch, i.e. one pass over `dataloader`.

    For every mini-batch: move the original images to `device`, run the
    model, compute the loss with `loss_function`, back-propagate, and
    update the parameters with `optimizer`. Use the original images as
    both inputs and reconstruction targets. Put the model in training
    mode first.
    Advance the optimizer's learning-rate scheduler after the epoch.

    Args:
        model:      the VAE, already on `device`.
        optimizer:  the optimizer from myoptim.get_optimizer.
        dataloader: yields batches of images, shape (B, 1, 32, 32), values
                    in [0, 1] (there are no labels).
        device:     torch.device to compute on.

    Returns:
        dict {"loss": ..., "recon": ..., "kl": ...} of Python floats: the
        average of each term per image over the whole epoch.
    """
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    total_images = 0

    for x in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon, kl = loss_function(x, x_hat, mu, logvar)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        for key, value in zip(totals, (loss, recon, kl)):
            totals[key] += value.item() * batch_size
        total_images += batch_size

    # The notebook owns the epoch loop, so advance the schedule here.
    scheduler = getattr(optimizer, "scheduler", None)
    if scheduler is not None:
        scheduler.step()

    return {key: value / total_images for key, value in totals.items()}
