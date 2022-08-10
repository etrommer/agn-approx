"""
Wrapper for LightningDataModule with boilerplate functionality
"""
import os

import pytorch_lightning as pl
import torch
import torch.utils.data as td


class ApproxDataModule(pl.LightningDataModule):
    """
    Superclass that provides a common dataloader boilerplate
    functionality for all datasets. Mostly derived from the Pytorch
    Lightning Docs.
    This class is not expected to be instantiated directly.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.args = kwargs
        self.data_dir = os.environ.get("AGNAPPROX_DATA_DIR", "./data")

    def _create_data_loader(self, data):
        return td.DataLoader(
            data,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
        )

    def train_dataloader(self):
        return self._create_data_loader(self.df_train)

    def val_dataloader(self):
        return self._create_data_loader(self.df_val)

    def test_dataloader(self):
        return self._create_data_loader(self.df_test)

    def sample_dataloader(self, num_samples=128):
        """
        Load a random sample from the training dataset.

        Args:
            num_samples: Number of samples to return. Defaults to 128.

        Returns:
            A dataloader instance with `num_samples` samples
        """
        indices = torch.randint(len(self.df_train), (num_samples,))
        return self._create_data_loader(td.Subset(self.df_train, indices))
