"""Offline-first datasets, two-view augmentation, and bounded replay memory."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5153, 0.4961, 0.4497), (0.2608, 0.2551, 0.2779)),
    "tiny-imagenet": ((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),
}


class TwoViews:
    """Apply the same augmentation pipeline independently twice."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class ProcessedTinyImageNet(Dataset):
    """Read the original repository's 20-shard Mammoth NumPy format."""

    def __init__(self, root, train, transform):
        folder = Path(root).expanduser() / "processed"
        split = "train" if train else "val"
        paths = [(folder / f"x_{split}_{i:02d}.npy", folder / f"y_{split}_{i:02d}.npy") for i in range(1, 21)]
        missing = [str(p) for pair in paths for p in pair if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing processed Tiny-ImageNet files, starting with {missing[0]}; see README")
        self.data = np.concatenate([np.load(x, allow_pickle=False) for x, _ in paths])
        self.targets = np.concatenate([np.load(y, allow_pickle=False) for _, y in paths]).reshape(-1)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(np.uint8(255 * self.data[index]))
        return self.transform(image), int(self.targets[index])


def image_transform(config, train, two_views=False):
    size = config.image_size
    operations = [transforms.Resize((size, size))]
    if train:
        operations += [
            transforms.RandomResizedCrop(size, scale=(0.1 if size > 32 else 0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0.5 if size > 32 else 0
            ),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(*STATS[config.dataset])]
    transform = transforms.Compose(operations)
    return TwoViews(transform) if two_views else transform


def load_dataset(config, root, train=True, two_views=False, download=False):
    root = str(Path(root).expanduser().resolve())
    transform = image_transform(config, train, two_views)
    if config.dataset == "tiny-imagenet":
        if download:
            raise ValueError("Prepare processed Tiny-ImageNet manually; automatic downloading is not supported")
        return ProcessedTinyImageNet(root, train, transform)
    cls = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
    try:
        return cls(root=root, train=train, transform=transform, download=download)
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot read {config.dataset} at {root}. Run `python -m gplasc download --dataset {config.dataset} --data-root {root}` on a networked machine first."
        ) from error


def indices_for_classes(targets, classes, per_class=None):
    targets = np.asarray(targets)
    parts = [np.flatnonzero(targets == label)[:per_class] for label in classes]
    return np.concatenate(parts).astype(np.int64).tolist() if parts else []


def update_memory(targets, previous, current, capacity):
    """Class-balanced random memory from prior memory and this task's data only."""
    candidates = np.asarray(list(previous) + list(current), dtype=np.int64)
    if capacity == 0 or len(candidates) == 0:
        return []
    targets = np.asarray(targets)
    classes = np.unique(targets[candidates])
    selected = []
    for class_index, label in enumerate(classes):
        pool = candidates[targets[candidates] == label]
        # Redistribute slots when a class has fewer samples than its fair share.
        remaining_classes = len(classes) - class_index
        quota = (capacity - len(selected)) // remaining_classes
        selected.extend(np.random.permutation(pool)[:quota].tolist())
    return selected


def make_loader(dataset, indices, config, train=True, balanced=False):
    if not indices:
        raise ValueError("Cannot construct a loader from an empty task")
    sampler = None
    if balanced:
        labels = np.asarray(dataset.targets)[indices]
        classes, counts = np.unique(labels, return_counts=True)
        frequencies = dict(zip(classes, counts))
        sampler = WeightedRandomSampler([1.0 / frequencies[label] for label in labels], len(indices))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
    )
