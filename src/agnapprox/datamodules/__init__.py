"""
Wrapper classes for common deep learning sets
using the pytorch-lightning DataModule
"""
from .approx_datamodule import ApproxDataModule
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .mnist import MNIST
from .tinyimagenet200 import TinyImageNet
