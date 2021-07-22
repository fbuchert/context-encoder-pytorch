import logging
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils.eval import AverageMeterSet
from utils.train import get_random_block_mask, get_random_region_mask, get_l2_weights


logger = logging.getLogger()


def evaluate(
        args,
        data_loader: DataLoader,
        generator: nn.Module,
        mask_size: int,
        overlap: int,
        global_pattern: Optional[torch.Tensor],
        reconstruction_loss,
        epoch: int,
        descriptor: str = "Test"
):
    meters = AverageMeterSet()

    generator.eval()

    if args.pbar:
        p_bar = tqdm(range(len(data_loader)))

    with torch.no_grad():
        for i, (samples, _) in enumerate(data_loader):
            samples = samples.to(args.device)
            if not args.random_masking:
                masked_samples, true_masked_part, mask_coordinates = get_random_block_mask(
                    samples, args.mask_size, args.overlap
                )
                masked_region = None
            else:
                masked_samples, masked_region = get_random_region_mask(
                    samples, args.image_size, args.mask_area, global_pattern, args.max_patter_size
                )
                true_masked_part = samples

            outG = generator(masked_samples)

            # Compute reconstruction / inpainting loss for generator
            l2_weights = get_l2_weights(args, outG.size(), masked_region)
            lossG_recon = reconstruction_loss(
                outG, true_masked_part, l2_weights.to(args.device)
            )

            meters.update("lossG_recon", lossG_recon.item(), 1)

            if args.pbar:
                p_bar.set_description(
                    "{descriptor}: Epoch: {epoch:4}. Iter: {batch:4}/{iter:4}. LossG Recon: {lossG_recon:.4f}.".format(
                        descriptor=descriptor,
                        epoch=epoch + 1,
                        batch=i + 1,
                        iter=len(data_loader),
                        lossG_recon=meters["lossG_recon"],
                    )
                )
                p_bar.update()
        p_bar.close()

    if not args.random_masking:
        recon_samples = samples.clone()
        h, w = mask_coordinates
        recon_samples[:, :, h: h + mask_size, w: w + mask_size] = outG
    else:
        recon_samples = outG

    recon_grid = make_grid(
        torch.cat([samples[:5], masked_samples[:5], recon_samples[:5]]),
        nrow=5,
        normalize=True,
    )
    return meters["lossG_recon"].avg, recon_grid
