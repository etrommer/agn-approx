"""
Wrapper for TinyImageNet dataset
"""
import os

import torch.utils.data as td
from torchvision import datasets, transforms

from .approx_datamodule import ApproxDataModule


class TinyImageNet(ApproxDataModule):
    """
    Dataloader instance for the TinyImageNet dataset
    """

    def __init__(self, split=0.9, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = os.path.join(self.data_dir, "tiny-imagenet-200")
        self.split = split

    @property
    def normalize(self):
        """
        Default ImageNet normalization parameters

        Returns:
            List of transformations to apply to input image
        """
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    @property
    def augment(self):
        """
        Default CIFAR10 augmentation pipeline

        Returns:
            List of transformations to apply to input image
        """
        return [transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)]

    def prepare_data(self):
        assert os.path.exists(self.data_dir), "Dataset not downloaded"

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            target_transform = transforms.Compose(self.augment + self.normalize)
            ds_full = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=target_transform
            )

            train_size = int(self.split * len(ds_full))
            val_size = len(ds_full) - train_size
            self.df_train, self.df_val = td.random_split(
                ds_full, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            target_transform = transforms.Compose(self.normalize)
            self.df_test = datasets.ImageFolder(
                os.path.join(self.data_dir, "val"), transform=target_transform
            )
