from typing import List, Tuple

from datasets.custom_datasets import *

DATASET_GETTERS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
    "svhn": datasets.SVHN,
    "stl10": STL10,
    "caltech101": Caltech101,
    "caltech256": Caltech256,
    "ham10000": HAM10000,
}


def get_datasets(
        root_dir: str,
        dataset: str,
        num_validation: float = 1,
        is_pct: bool = False,
        train_transform: Callable = None,
        test_transform: Callable = None,
        download: bool = True,
        dataset_indices: Optional[List] = None
):
    """
    Method that returns all dataset objects required for semi-supervised learning: labeled train set, unlabeled train
    set, validation set and test set. The returned dataset objects can be used as input for data loaders used during
    model training.

    Parameters
    ----------
    root_dir: str
        Path to root data directory to load datasets from or download them to, if not downloaded yet, e.g. `./data`.
    dataset: str
        Name of dataset, e.g. `cifar10`, `imagenet`, etc.
    num_validation: float
        Number of samples selected for the validation set. These samples are selected from all available
        training samples.
    is_pct: bool (default: False)
        Flag indicating whether num_labeled / num_validation are given in pct of the base_set or in absolute numbers.
    train_transform: Callable
        Transform / augmentation strategy applied to the labeled training set.
    test_transform: Callable
        Transform / augmentation strategy applied to the validation and test set.
    download: bool
        Boolean indicating whether the dataset should be downloaded or not. If yes, the get_base_sets method will
        download the dataset to the root dir if possible. Automatic downloading is supported for CIFAR-10, CIFAR-100,
        STL-10 and ImageNet.
    dataset_indices: Optional[Dict]
        Dictionary containing indices for the labeled and unlabeled training sets, validation set and test set for
        initialization. This argument should be used if training is resumed, i.e. initializing the dataset splits to
        the same indices as in the previous training run, and dataset indices are loaded. An alternative use case,
        would be to select initial indices in a principled way, e.g. selecting diverse initial samples based on
        representations provided by self-supervised learning.
    Returns
    -------
    dataset_tuple: Tuple[Dict, List, List]
        Returns tuple containing dataset objects of all relevant datasets. The first tuple element is a dictionary
        containing the labeled training dataset at key `labeled` and the unlabeled training dataset at key unlabeled.
        The second and third elements are the validation dataset and the test dataset.
    """
    base_set, test_set = get_base_sets(
        dataset, root_dir, download=download, test_transform=test_transform
    )

    if is_pct:
        num_validation = int(num_validation * len(base_set))

    base_indices = list(range(len(base_set)))
    if dataset_indices is None:
        num_training = len(base_indices) - num_validation
        train_indices, validation_indices = get_uniform_split(base_set.targets, base_indices, split_num=num_training)
    else:
        train_indices, validation_indices = dataset_indices["train_labeled"], dataset_indices["validation"]

    train_set = CustomSubset(
        base_set, train_indices, transform=train_transform
    )
    validation_set = CustomSubset(
        base_set, validation_indices, transform=test_transform
    )

    return train_set, validation_set, test_set


def get_base_sets(dataset, root_dir, download=True, test_transform=None):
    base_set = DATASET_GETTERS[dataset](root_dir, train=True, download=download)
    test_set = DATASET_GETTERS[dataset](
        root_dir, train=False, download=download, transform=test_transform
    )
    return base_set, test_set


def get_uniform_split(targets: List, indices: List, split_pct: float = None, split_num: int = None):
    """
    Method that splits provided train_indices uniformly according to targets / class labels, i.e. it returns a random
    split of train_indices s.t. indices in both splits are ideally uniformly distributed among classes (as done
    in MixMatch implementation by default).

    Parameters
    ----------
    indices: List
        List of dataset indices on which split should be performed.
    targets: List
        List of targets / class labels corresponding to provided indices of dataset. Based on the provided targets,
        the indices are split s.t. samples in split0 and split1 are uniformly distributed among classes as well as
        possible.
    split_num: int
        Number of total samples selected for first split. Alternatively one can specify a split percentage by providing
        split_pct as input.
    split_pct: float
        Percentage of all indices which are selected for the first split. Should only specified if split_num is not given.
    Returns
    ----------
    split_indices: Tuple[List, List]
        Returns two lists, which contain the indices split according to the parameters split_num or split_pct.
    """
    if split_pct is not None:
        samples_per_class = (split_pct * len(indices)) // len(np.unqiue(targets))
    elif split_num is not None:
        samples_per_class = split_num // len(np.unique(targets))
    else:
        raise ValueError('Expected either split_pct or split_num to be not None.')

    split0_indices, split1_indices = [], []
    for class_label in np.unique(targets):
        class_indices = np.where(np.array(targets)[indices] == class_label)[0]
        np.random.shuffle(class_indices)
        split0_indices += list(class_indices[:samples_per_class])
        split1_indices += list(class_indices[samples_per_class:])
    split0_indices = np.array(indices)[split0_indices].tolist()
    split1_indices = np.array(indices)[split1_indices].tolist()

    # Make sure that the number of selected indices exactly matches split_num if not None
    # If this is not the case, randomly sample indices from split1 and add them to split0
    if split_num is not None and len(split0_indices) < split_num:
        tmp_indices = random.sample(split1_indices, split_num - len(split0_indices))
        split0_indices += tmp_indices
        split1_indices = np.setdiff1d(split1_indices, tmp_indices).tolist()
    return split0_indices, split1_indices
