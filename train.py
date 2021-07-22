import argparse
import os
import logging
from tqdm import tqdm
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torchvision import transforms
from torchvision.utils import make_grid

from augmentation.augmentations import get_normalizer
from eval import evaluate
from utils.eval import AverageMeterSet
from utils.misc import save_state
from utils.train import get_l2_weights, get_random_block_mask, get_random_region_mask, context_encoder_init
from models.model_factory import MODEL_GETTERS

GLOBAL_RANDOM_PATTERN = None

logger = logging.getLogger()


def get_transform_dict(args):
    """
    Generates dictionary with transforms for all datasets. The context encoder just uses normalized images.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object that contains all command line arguments with their corresponding values
    Returns
    -------
    transform_dict: Dict
        Dictionary containing transforms for the labeled train set, unlabeled train set
        and the validation / test set
    """
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            get_normalizer(args.dataset)
        ]
    )
    return {"train": transform, "train_unlabeled": None, "test": transform}


def get_optimizer(args, model):
    """
    Initialize and return Adam optimizer

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values.
    model: torch.nn.Module
        torch module which is trained using MixMatch.
    Returns
    -------
    optim: torch.optim.Optimizer
        Returns adam optimizer which is used for model training.
    """
    return Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


def get_scheduler(args, optimizer):
    return None


def weighted_mse_loss(outputs, targets, weights):
    return torch.mean(weights * (outputs - targets).pow(2))


def train(args, train_loader, validation_loader, test_loader, writer, **kwargs):
    """
    Method for ContextEncoder training of model based on given data loaders and parameters.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values.
    train_loader: DataLoader
        Data loader of train set.
    validation_loader: DataLoader
        Data loader of validation set (usually empty).
    test_loader: DataLoader
        Data loader of test set.
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    Returns
    -------
    generator: torch.nn.Module
        Returns the trained ContextEncoder generator model.
    discriminator: torch.nn.Module
        Returns the trained ContextEncoder discriminator model.
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    """
    save_path = kwargs.get("save_path", args.out_dir)
    args.mask_size = int(np.sqrt(args.mask_area) * args.image_size)

    if args.random_masking:
        global GLOBAL_RANDOM_PATTERN
        GLOBAL_RANDOM_PATTERN = generate_random_pattern(args.mask_area, args.resolution, args.max_pattern_size)
        out_size = args.image_size
    else:
        out_size = args.mask_size

    # Instantiate generator and discriminator
    generator = MODEL_GETTERS["context_generator"](
        bottleneck_dim=args.bottleneck, img_size=args.image_size, out_size=out_size
    )
    discriminator = MODEL_GETTERS["context_discriminator"](
        bottleneck_dim=args.bottleneck, input_size=out_size
    )

    # Instantiate generator and discriminator
    generator.apply(context_encoder_init)
    generator.to(args.device)
    discriminator.apply(context_encoder_init)
    discriminator.to(args.device)

    optim_g = get_optimizer(args, generator)
    optim_d = get_optimizer(args, discriminator)

    adversarial_loss = nn.BCELoss().to(args.device)
    reconstruction_loss = weighted_mse_loss

    for epoch in range(args.epochs):
        lossG_total, lossG_recon, lossG_adv, lossD_total, train_recon_grid = train_epoch(
            args,
            generator,
            discriminator,
            train_loader,
            optim_g,
            optim_d,
            reconstruction_loss,
            adversarial_loss,
            epoch,
        )

        test_lossG_total, test_recon_grid = evaluate(
            args,
            test_loader,
            generator,
            args.mask_size,
            args.overlap,
            GLOBAL_RANDOM_PATTERN,
            reconstruction_loss,
            epoch,
            descriptor="Test",
        )

        writer.add_scalar("Loss/train_lossG_total", lossG_total, epoch)
        writer.add_scalar("Loss/train_lossG_recon", lossG_recon, epoch)
        writer.add_scalar("Loss/train_lossG_adv", lossG_adv, epoch)
        writer.add_scalar("Loss/train_lossD_total", lossD_total, epoch)
        writer.add_scalar("Loss/testt_lossG_total", test_lossG_total, epoch)
        writer.add_image("train_reconstructions", train_recon_grid, epoch)
        writer.add_image("test_reconstructions", test_recon_grid, epoch)
        writer.flush()

        if epoch % args.checkpoint_interval == 0 and args.save:
            save_state(epoch, generator, discriminator, optim_g, optim_d, save_path, "checkpoint_{}.tar".format(epoch))

    writer.close()
    save_state(epoch, generator, discriminator, optim_g, optim_d, save_path, "last_model.tar")
    return generator, discriminator, writer


def train_epoch(
        args: argparse.Namespace,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        train_loader: DataLoader,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        reconstruction_loss: Callable,
        adversarial_loss: Callable,
        epoch: int
):
    """
    Method that executes a training epoch, i.e. a pass through all train samples in the training data loaders.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    generator: torch.nn.Module
        Model, i.e. neural network to train using MixMatch.
    discriminator: torch.nn.Module
        The EMA class which maintains an exponential moving average of model parameters. In MixMatch the exponential
        moving average parameters are used for model evaluation and for the reported results.
    train_loader: DataLoader
        Data loader fetching batches from the labeled set of data.
    optim_g: Optimizer
        Optimizer object for the generator model
    optim_d: Optimizer
        Optimizer for the discriminator model
    reconstruction_loss: Callable
        Reconstruction loss computed w.r.t input images and generator output.
    adversarial_loss: Callable
        Adversarial loss computed based on real and fake input images.
    epoch: int
        Current epoch
    Returns
    -------
    train_stats: Tuple
        The method returns a tuple containing the total, labeled and unlabeled loss.
    """
    meters = AverageMeterSet()

    generator.zero_grad()
    generator.train()
    discriminator.zero_grad()
    discriminator.train()

    real_labels = torch.ones(args.batch_size).to(args.device)
    fake_labels = torch.zeros(args.batch_size).to(args.device)

    if args.pbar:
        p_bar = tqdm(range(len(train_loader)))

    for batch_idx, (samples, _) in enumerate(train_loader):
        samples = samples.to(args.device)
        if not args.random_masking:
            masked_samples, true_masked_part, mask_coordinates = get_random_block_mask(
                samples, args.mask_size, args.overlap
            )
            masked_region = None
        else:
            masked_samples, masked_region = get_random_region_mask(
                samples, args.image_size, args.mask_area, GLOBAL_RANDOM_PATTERN
            )
            true_masked_part = samples

        # ------------------------------------------
        # Update discriminator
        # ------------------------------------------
        # Compute adversarial loss for discriminator loss on real samples
        discriminator.zero_grad()
        outD_real = discriminator(true_masked_part)
        lossD_real = adversarial_loss(outD_real, real_labels)

        outG = generator(masked_samples)

        # Compute adversarial loss for discriminator on fake samples generated by generator
        outD_fake = discriminator(outG.detach())
        lossD_fake = adversarial_loss(outD_fake, fake_labels)

        lossD_total = (lossD_real + lossD_fake) * 0.5
        lossD_total.backward()
        optim_d.step()

        # ------------------------------------------
        # Update generator
        # ------------------------------------------
        generator.zero_grad()

        # Compute adversarial loss for generator
        outD_fake = discriminator(outG)
        # "real" labels as generator tries to fool discriminator
        lossG_fake = adversarial_loss(outD_fake, real_labels)

        # Compute reconstruction / inpainting loss for generator
        l2_weights = get_l2_weights(args, outG.size(), masked_region)
        lossG_recon = reconstruction_loss(
            outG, true_masked_part, l2_weights.to(args.device)
        )
        lossG_total = (1 - args.w_rec) * lossG_fake + args.w_rec * lossG_recon

        lossG_total.backward()
        optim_g.step()

        meters.update("lossG_adv", lossG_fake.item(), 1)
        meters.update("lossG_recon", lossG_recon.item(), 1)
        meters.update("lossG_total", lossG_total.item(), 1)
        meters.update("lossD_total", lossD_total.item(), 1)

        if args.pbar:
            p_bar.set_description(
                "Train Epoch: {epoch:4}/{total_epochs:4}. Iter: {batch:4}/{iter:4}. LossG: {lossG_total:.4f}. "
                "LossD Total: {lossD_total:.4f}.".format(
                    epoch=epoch + 1,
                    total_epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader),
                    lossG_total=meters["lossG_total"],
                    lossG_recon=meters["lossG_recon"],
                    lossG_adv=meters["lossG_adv"],
                    lossD_total=meters["lossD_total"],
                )
            )
            p_bar.update()

    if args.pbar:
        p_bar.close()

    if not args.random_masking:
        recon_samples = samples.clone()
        h, w = mask_coordinates
        recon_samples[:, :, h : h + args.mask_size, w : w + args.mask_size] = outG
    else:
        recon_samples = outG

    recon_grid = make_grid(
        torch.cat([samples[:5], masked_samples[:5], recon_samples[:5]]),
        nrow=5,
        normalize=True,
    )
    return (
        meters["lossG_total"].avg,
        meters["lossG_recon"].avg,
        meters["lossG_adv"].avg,
        meters["lossD_total"].avg,
        recon_grid,
    )


def generate_random_pattern(mask_area: float, resolution: float, max_pattern_size: int):
    """
    Generates global random pattern based on which random region masks can be sampled.
    TODO: Add reference
    """
    pattern = torch.rand((int(resolution * max_pattern_size), int(resolution * max_pattern_size))).multiply_(255)
    resized_pattern = F.interpolate(pattern[None, None, :, :], max_pattern_size, mode="bicubic", align_corners=False)
    resized_pattern = resized_pattern.squeeze().div_(255)
    return torch.lt(resized_pattern, mask_area).bool()
