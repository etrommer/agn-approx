"""
Wrapper for CIFAR10 dataset
"""
import torch.utils.data as td
from agnapprox.datamodules import ApproxDataModule
from torchvision import datasets, transforms


class CIFAR10(ApproxDataModule):
    """
    Dataloader instance for the CIFAR10 dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def normalize(self):
        """
        Default CIFAR10 normalization parameters

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
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            target_transform = transforms.Compose(self.augment + self.normalize)
            cifar_full = datasets.CIFAR10(
                root=self.data_dir, train=True, transform=target_transform
            )
            self.df_train, self.df_val = td.random_split(cifar_full, [45000, 5000])
        if stage == "test" or stage is None:
            target_transform = transforms.Compose(self.normalize)
            self.df_test = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=target_transform
            )
