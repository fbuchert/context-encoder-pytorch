import argparse


def parse_args():
    """
    parse_args parses command line arguments and returns argparse.Namespace object

    Returns
    -------
    argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    """
    parser = argparse.ArgumentParser(description="Context encoder training")

    # General arguments
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        choices=["cpu", "cuda"],
        help="Device used for training",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="Number of workers used for data loading",
    )
    parser.add_argument(
        "--out-dir",
        default="./out",
        type=str,
        help="path to which output logs, losses and model checkpoints are saved.",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "imagenet",
            "svhn",
            "caltech101",
            "caltech256",
            "stl10",
            "ham10000",
        ],
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--data-dir", default="./data", type=str, help="path to which dataset is saved"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to checkpoint from which to resume training",
    )
    parser.add_argument("--epochs", default=1024, type=int, help="number of epochs")
    parser.add_argument(
        "--iters-per-epoch",
        default=1024,
        type=int,
        help="number of iterations per epoch",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="batch_size")
    parser.add_argument("--lr", default=0.03, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=0.0005, type=float, help="weight decay")
    parser.add_argument(
        "--ema-decay",
        default=0.999,
        type=float,
        help="exponential moving average decay",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Should CPU tensors be directly allocated in Pinned memory for data loading",
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=500,
        help="Number of initially labeled samples",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Interval [epoch] in which checkpoints are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=65,
        help="Manually set seed for random number generators",
    )
    parser.add_argument(
        "--trainable-layers",
        type=str,
        nargs='+',
        default=[],
        help='If pretrained flag is set, this specifies the layers for which weights should be frozen'
    )

    # Flags
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=False,
        help="Flag indicating if models should be loaded as pretrained (if available) or not",
    )
    parser.add_argument(
        "--weighted-sampling",
        dest="weighted_sampling",
        action="store_true",
        default=False,
        help="""Flag indicating if batches selects samples inversely proportional to class distribution,
                i.e. whether on average samples from each class should be selected with equal probability"""
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=False,
        help="Flag indicating if models should be saved or not",
    )
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store_true",
        default=False,
        help="if specified seed is randomly set",
    )
    parser.add_argument(
        "--polyaxon",
        dest="polyaxon",
        action="store_true",
        default=False,
        help="Flag indicating if training is run on polyaxon or not",
    )
    parser.add_argument(
        "--pbar", dest="pbar", action="store_true", default=False, help="print progress bar"
    )

    # Dataset split settings
    parser.add_argument("--initial-indices", type=str, help='path to initial indice file to start from')
    parser.add_argument(
        "--num-validation",
        type=float,
        default=1,
        help="Specify number of samples in the validation set",
    )
    parser.add_argument(
        "--is-pct", dest="is_pct", action="store_true", default=False,
        help="Flag specifying whether --num-validation are given in percent or not."
    )

    # Context encoder arguments
    parser.add_argument(
        "--random-masking",
        dest="random_masking",
        action="store_true",
        default=False,
        help="apply random region masking during training",
    )
    parser.add_argument(
        "--bottleneck",
        type=int,
        default=2048,
        help="dimension of bottleneck layer of context generator",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        choices=[32, 64, 128],
        help="size of input images",
    )
    parser.add_argument(
        "--w-rec", type=float, default=0.999, help="weight of reconstruction error"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 of adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 of adam optimzer"
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="overlap pixels applied in masking"
    )
    parser.add_argument(
        "--overlap-weight-multiplier",
        type=int,
        default=10,
        help="weight multiplier for reconstruction loss on overlap pixels",
    )
    parser.add_argument(
        "--mask-area",
        type=float,
        default=0.25,
        help="area which is masked during training",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.03,
        help="Resolution of global pattern used in case of random pattern masking (as opposed to masking using squared"
             "blocks"
    )
    parser.add_argument(
        "--max-pattern-size",
        type=int,
        default=10000,
        help="Size of randomly selected pattern (w.r.t to global pattern)"
    )

    return parser.parse_args()
