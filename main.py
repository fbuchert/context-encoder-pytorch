import os
import json
import logging

from torch.utils.tensorboard import SummaryWriter
from polyaxon_client.tracking import get_data_paths, get_outputs_path

from arguments import parse_args
from train import get_transform_dict, train
from datasets.datasets import get_datasets
from datasets.loaders import create_loaders
from utils.train import model_init, set_grads
from utils.misc import load_dataset_indices, save_dataset_indices, get_save_path, initialize_logger, seed, save_args
from models.model_factory import MODEL_GETTERS


logger = logging.getLogger()


def main(args, save_path: str):
    """
    Main function that sets up and starts the context encoder training
    """
    writer = SummaryWriter(save_path)

    # Load initial dataset from path specified by args.resume / args.initial_indices if set
    initial_indices = None
    if args.resume:
        initial_indices = load_dataset_indices(args.resume)
    elif args.initial_indices:
        path, file_name = os.path.split(args.initial_indices)
        initial_indices = load_dataset_indices(path, file_name)

    # Get dictionary which contains train transforms (both for labeled and unlabeled batches) as well as
    # the transform for the validation and test set
    transform_dict = get_transform_dict(args)

    # Get torch dataset objects from specified dataset
    train_set, validation_set, test_set = get_datasets(
        args.data_dir,
        args.dataset,
        args.num_validation,
        args.is_pct,
        transform_dict["train"],
        transform_dict["test"],
        dataset_indices=initial_indices
    )
    save_dataset_indices(save_path, train_set, validation_set)

    # Get loaders for the labeled and unlabeled train set as well as the validation and test set
    args.iters_per_epoch = (len(train_set) // args.batch_size) + 1
    train_loader, validation_loader, test_loader = create_loaders(
        args,
        train_set,
        validation_set,
        test_set,
        args.batch_size,
        total_iters=args.iters_per_epoch,
        num_workers=args.num_workers
    )

    # Print and log dataset stats
    logger.info("-------- Starting Unsupervised Context Encoder Training --------")
    logger.info("\t- Train set: {}".format(len(train_set)))
    logger.info("\t- Validation set: {}".format(len(validation_set)))
    logger.info("\t- Test set: {}".format(len(test_set)))

    # Start context encoder training
    train(
        args,
        train_loader,
        validation_loader,
        test_loader,
        writer,
        save_path=save_path
    )
    save_args(args, save_path)


if __name__ == '__main__':
    # Read command line arguments
    args = parse_args()

    # Set up paths if code is run as polyaxon experiment
    if args.polyaxon:
        args.out_dir = os.path.join(get_outputs_path(), args.out_dir)
        if args.initial_model:
            args.initial_model = os.path.join(get_data_paths()['data1'], args.initial_model)
            args.seed = json.load(open(os.path.join(args.initial_model, "args.json")))[
                "seed"
            ]
        elif args.resume:
            args.resume = os.path.join(get_data_paths()["data1"], args.resume)
    save_path = get_save_path(args)

    initialize_logger(save_path)
    args.seed = seed(args.random_seed, args.seed)
    logger.info("Seed is set to {}".format(args.seed))
    main(args, save_path)
