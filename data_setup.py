"""This function creates PyTorch DataLoaders for iamge classification data"""

import os
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """
    Args:
    train_dir: path to training directory
    test_dir: path to testing directory
    transforms: data augmentation for training and testing data
    batch_size: number of samples per batch in each DataLoader
    num_workers: number of workers in each DataLoader

    Returns:
    Tuple of (train_dataloader, test_dataloader, class_names)
    class_names: list of target classes
    """

    # use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)

    # get class names
    class_names = train_data.classes

    # Turn Images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # do not shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


def split_dataset(
    dataset: torchvision.datasets, split_size: float = 0.2, seed: int = 42
):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be the validation set? Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and
            random_split_2 is of size (1-split_size)*len(dataset).
    """

    # create split length based on original dataset size
    size_1 = int(len(dataset) * split_size)  # desired length
    size_2 = len(dataset) - size_1  # remaining length

    print(
        f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {size_1} ({int(split_size * 100)}%), {size_2} ({int((1 - split_size) * 100)}%)"
    )

    # create splits given random seed
    set_1, set_2 = torch.utils.data.random_split(
        dataset, lengths=[size_1, size_2], generator=torch.manual_seed(seed)
    )

    return set_1, set_2
