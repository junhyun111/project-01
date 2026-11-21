import math
import torch
import torch.nn.functional as F

from myoptim import learning_rate_for_epoch


def _affine_strength(epoch):
    if epoch <= 300:
        factor = 1.0
    elif epoch <= 450:
        factor = (
            (450 - epoch) / 150.0
        )
    else:
        factor = 0.0

    return {
        "p": 0.50 * factor,
        "rot": 8.0 * factor,
        "scale": 0.10 * factor,
        "shift_px": 1.0 * factor,
        "shear": 0.10 * factor,
    }


def _augment(x, epoch):
    cfg = _affine_strength(epoch)

    if cfg["p"] <= 0:
        return x

    b = x.shape[0]
    device = x.device

    apply = (
        torch.rand(b, device=device)
        < cfg["p"]
    )

    angle = (
        (torch.rand(b, device=device) * 2 - 1)
        * cfg["rot"]
        * math.pi
        / 180.0
    )

    scale = (
        1.0
        + (
            torch.rand(
                b, device=device
            ) * 2 - 1
        ) * cfg["scale"]
    )

    shear = (
        (
            torch.rand(
                b, device=device
            ) * 2 - 1
        ) * cfg["shear"]
    )

    tx = (
        (
            torch.rand(
                b, device=device
            ) * 2 - 1
        )
        * cfg["shift_px"]
        * (2.0 / x.shape[-1])
    )

    ty = (
        (
            torch.rand(
                b, device=device
            ) * 2 - 1
        )
        * cfg["shift_px"]
        * (2.0 / x.shape[-2])
    )

    theta = torch.zeros(
        b, 2, 3,
        device=device,
        dtype=x.dtype,
    )

    theta[:, 0, 0] = (
        torch.cos(angle) / scale
    )
    theta[:, 0, 1] = (
        -torch.sin(angle) + shear
    ) / scale
    theta[:, 1, 0] = (
        torch.sin(angle) / scale
    )
    theta[:, 1, 1] = (
        torch.cos(angle) / scale
    )
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(
        theta,
        x.shape,
        align_corners=False,
    )

    x_aug = F.grid_sample(
        x,
        grid,
        align_corners=False,
        padding_mode="zeros",
    ).clamp(0, 1)

    return torch.where(
        apply[:, None, None, None],
        x_aug,
        x,
    )


def loss_function(
    x,
    x_hat,
    mu,
    logvar,
):
    batch_size = x.size(0)

    recon = F.binary_cross_entropy(
        x_hat,
        x,
        reduction="sum",
    ) / batch_size

    kl = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    ).sum() / batch_size

    return recon + kl, recon, kl


def train_one_epoch(
    model,
    optimizer,
    dataloader,
    device,
):
    model.train()

    epoch = (
        getattr(
            optimizer,
            "vae_epoch",
            0,
        )
        + 1
    )

    lr = learning_rate_for_epoch(
        epoch
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
    }

    total_images = 0

    for x in dataloader:
        x = x.to(device)
        x = _augment(x, epoch)

        optimizer.zero_grad()

        x_hat, mu, logvar = model(x)

        loss, recon, kl = loss_function(
            x,
            x_hat,
            mu,
            logvar,
        )

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        totals["loss"] += (
            loss.item() * batch_size
        )
        totals["recon"] += (
            recon.item() * batch_size
        )
        totals["kl"] += (
            kl.item() * batch_size
        )

        total_images += batch_size

    optimizer.vae_epoch = epoch

    return {
        key: value / total_images
        for key, value in totals.items()
    }
