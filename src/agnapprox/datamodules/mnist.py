"""
Wrapper for MNIST dataset
"""
import torch.utils.data as td
from torchvision import datasets, transforms

from .approx_datamodule import ApproxDataModule


class MNIST(ApproxDataModule):
    """
    Dataloader instance for the MNIST dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def normalize(self):
        """
        Default MNIST normalization pipeline

        Returns:
            List of transformations to apply to input image
        """
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            target_transform = transforms.Compose(self.normalize)
            mnist_full = datasets.MNIST(
                root=self.data_dir, train=True, transform=target_transform
            )
            self.df_train, self.df_val = td.random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            target_transform = transforms.Compose(self.normalize)
            self.df_test = datasets.MNIST(
                root=self.data_dir, train=False, transform=target_transform
            )
