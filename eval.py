import argparse
import logging

import torchvision.utils
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
                    samples, args.image_size, args.mask_area, global_pattern
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


def parse_args():
    parser = argparse.ArgumentParser(description='Context Encoder evaluation')

    parser.add_argument('--run-path', type=str, help='path to context encoder run which should be evaluated.')
    parser.add_argument('--data-dir', default='./data', type=str, help='path to directory where datasets are saved.')
    parser.add_argument('--save-path', type=str, help='path to which grid of reconstructed test images is saved.')
    parser.add_argument('--checkpoint-file', default='', type=str, help='name of .tar-checkpoint file from which model is loaded for evaluation.')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'gpu'], help='device (cpu / cuda) on which evaluation is run.')
    parser.add_argument('--pbar', action='store_true', default=False, help='flag indicating whether or not to show progress bar for evaluation.')
    return parser.parse_args()


if __name__ == '__main__':
    import os
    from train import generate_random_pattern, weighted_mse_loss
    from utils.misc import load_args, load_state
    from augmentation.augmentations import get_normalizer
    from datasets.datasets import get_base_sets
    from models.model_factory import MODEL_GETTERS

    args = parse_args()
    args.device = torch.device(args.device)
    args.save_path = args.save_path if args.save_path else args.run_path

    # Load arguments of run to evaluate
    run_args = load_args(args.run_path)

    # Initialize test dataset and loader
    _, test_set = get_base_sets(run_args.dataset, args.data_dir, test_transform=get_normalizer(run_args.dataset))
    test_loader = DataLoader(
        test_set,
        batch_size=run_args.batch_size,
        num_workers=run_args.num_workers,
        shuffle=False,
        pin_memory=run_args.pin_memory,
    )

    if run_args.random_masking:
        out_size = run_args.image_size
    else:
        out_size = run_args.mask_size

    # Load trained model from specified checkpoint .tar-file containing model state dict
    generator = MODEL_GETTERS["context_generator"](
        bottleneck_dim=run_args.bottleneck, img_size=run_args.image_size, out_size=out_size
    )

    if args.checkpoint_file:
        saved_state = load_state(os.path.join(args.run_path, args.checkpoint_file), map_location=args.device)
    else:
        checkpoint_file = next(filter(lambda x: x.endswith('.tar'), sorted(os.listdir(args.run_path), reverse=True)))
        saved_state = load_state(os.path.join(args.run_path, checkpoint_file), map_location=args.device)

    generator.load_state_dict(saved_state['generator_state_dict'])
    random_pattern = generate_random_pattern(run_args.mask_area, run_args.resolution, run_args.max_pattern_size)
    lossG_recon, recon_grid = evaluate(
        run_args,
        test_loader,
        generator,
        run_args.mask_size,
        run_args.overlap,
        random_pattern,
        weighted_mse_loss,
        saved_state['epoch']
    )
    grid_save_path = os.path.join(args.save_path, 'test_image_recon_grid.png')

    print(' CONTEXT ENCODER EVALUATION '.center(50, '-'))
    print(f'\t - Dataset {run_args.dataset}')
    print(f'\t - Test metrics:')
    print(f'\t\t Reconstruction loss: {lossG_recon}')
    print(f'Saving test reconstruction grid to {grid_save_path}.')
    torchvision.utils.save_image(recon_grid, grid_save_path)
