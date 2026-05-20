import random
import numpy as np
import torch
import lightning as L
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed, workers=True)


def build_transform(dataset_name: str, train: bool = True):
    """
    Возвращает transform для конкретного датасета.
    Для каждого датасета свои mean/std и аугментации.
    """
    if dataset_name.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name.lower() == "cifar100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif dataset_name.lower() == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if train:
        if dataset_name.lower() in ("cifar10", "cifar100"):
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.TrivialAugmentWide(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif dataset_name.lower() == "svhn":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transform


class TransformedSubset(torch.utils.data.Dataset):
    """Обёртка для Subset, применяющая transform."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def make_cutmix_mixup_collate(num_classes: int):
    """Возвращает collate-функцию с CutMix / MixUp."""
    cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
    mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)
    choice = v2.RandomChoice([cutmix, mixup])

    def collate(batch):
        return choice(*default_collate(batch))

    return collate
