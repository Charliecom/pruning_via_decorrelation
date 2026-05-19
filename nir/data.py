import lightning as L
from torch.utils.data import DataLoader, random_split
import torchvision as tv
from myutils import TransformedSubset, build_transform, make_cutmix_mixup_collate


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar100",
        data_path: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset.lower()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.num_classes = self._get_num_classes()

    def _get_num_classes(self):
        if self.dataset_name == "cifar10":
            return 10
        elif self.dataset_name == "cifar100":
            return 100
        elif self.dataset_name == "svhn":
            return 10
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def setup(self, stage=None):
        test_transform = build_transform(self.dataset_name, train=False)
        train_transform = build_transform(self.dataset_name, train=True)
        val_transform = build_transform(self.dataset_name, train=False)

        # Загружаем train (без transform) и test (с transform)
        if self.dataset_name == "cifar10":
            train_full = tv.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=None
            )
            self.test_data = tv.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=test_transform
            )
        elif self.dataset_name == "cifar100":
            train_full = tv.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=None
            )
            self.test_data = tv.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=test_transform
            )
        elif self.dataset_name == "svhn":
            train_full = tv.datasets.SVHN(
                root=self.data_dir, split="train", download=True, transform=None
            )
            self.test_data = tv.datasets.SVHN(
                root=self.data_dir,
                split="test",
                download=True,
                transform=test_transform,
            )

        # Разбиваем train на train/val
        val_size = int(len(train_full) * self.val_split)
        train_size = len(train_full) - val_size
        train_sub, val_sub = random_split(train_full, [train_size, val_size])

        self.train_data = TransformedSubset(train_sub, train_transform)
        self.val_data = TransformedSubset(val_sub, val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=make_cutmix_mixup_collate(self.num_classes),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
