<p align="center">
<img width="400" height="400" src="https://raw.githubusercontent.com/etrommer/agn-approx/main/docs/agnapprox_logo.png" alt="AGN Approx Logo">
</p>

# AGN Approx
Code and experiments for the paper [Combining Gradients and Probabilities for Heterogeneours Approximation of Neural Networks](https://arxiv.org/abs/2208.07265).
`agnapprox` allows for the study of neural networks using [Approximate Multipliers](https://en.wikipedia.org/wiki/Approximate_computing). It's main purpose is to optimize the _assignment_ of different approximate multipliers to the different layers of a Neural Network.
By learning a perturbation term for each layer, agnapprox finds out which layers are more or less resilient to small errors in the computations. This information is then used to choose accurate/inaccurate approximate multipliers for each layer.
The documentation contains two tutorials on agnapprox' functionality and demonstrates how to optimize a neural network supplied by the user.

## Documentation
Detailed Documentation can be found under: [https://etrommer.github.io/agn-approx/](https://etrommer.github.io/agn-approx/)

## Installation
This project is not yet hosted on PyPi. You can install it directly from this repository using `pip`:
```bash
$ pip install git+https://github.com/etrommer/agn-approx.git
```
### Code Formatting & Linting (Development only)
To automatically set up [pre-commit](https://pre-commit.com/), run:
```bash
poetry run pre-commit install
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
- `agnapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
- This work was created as part of my Ph.D. research at Infineon Technologies Dresden
