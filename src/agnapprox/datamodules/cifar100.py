"""
Wrapper for CIFAR10 dataset
"""
import torch.utils.data as td
from torchvision import datasets, transforms

from .cifar10 import CIFAR10


class CIFAR100(CIFAR10):
    """
    Dataloader instance for the CIFAR10 dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(self):
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            target_transform = transforms.Compose(self.augment + self.normalize)
            cifar_full = datasets.CIFAR100(
                root=self.data_dir, train=True, transform=target_transform
            )
            self.df_train, self.df_val = td.random_split(cifar_full, [45000, 5000])
        if stage == "test" or stage is None:
            target_transform = transforms.Compose(self.normalize)
            self.df_test = datasets.CIFAR100(
                root=self.data_dir, train=False, transform=target_transform
            )
