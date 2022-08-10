# AGN Approx

Code and experiments for the paper "Combining Gradients and Probabilities for Heterogeneours Approximation of Neural Networks"
This is currently a work in progress.

## Note
*This package relies on the Python package TorchApprox for GPU-accelerated layer implementations. This package is currently not publicly available. It will likely be made available in late 2022/early 2023. If you need early access, please get in touch*

## Installation
This project is not yet hosted on PyPi. You can install it directly from this repository using `pip`:
```bash
$ pip install git+https://github.com/etrommer/agn-approx.git
```
### Tiny ImageNet 200
Different from CIFAR10 and MNIST which are available through `torchvision`, the Tiny ImageNet dataset needs to be downloaded manually:
```bash
$ cd <your data dir>
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
$ unzip tiny-imagenet-200.zip
```
The validation images are provided in a flat folder with labels contained in a separate text file. This needs to be changed to a folder structure where each folder is a class containing the respective images. There is a script that handles the conversion:
```bash
$ ./src/agnapprox/datamodules/format_tinyimagenet.py --path <your data dir>/tiny-imagenet-200
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`agnapprox` was created by Elias Trommer. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`agnapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
