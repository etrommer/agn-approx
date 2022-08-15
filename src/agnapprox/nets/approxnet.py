"""
Approximate Neural Network boilerplate implementation
"""
# pylint: disable=arguments-differ
import logging
from typing import List, Optional, Tuple, Type

import agnapprox.utils
import mlflow
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchapprox.layers as al
from evoapproxlib import EvoApproxLib

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class ApproxNet(pl.LightningModule):
    """
    Base Class that provideds common functionality for approximate neural network training
    """

    def __init__(self, deterministic: bool = False):
        super().__init__()

        self._mode: str = "baseline"
        self._total_ops: Optional[int] = None
        self.lmbd: float = 0.0
        self.sigma_max: float = 0.5
        self.deterministic: bool = deterministic
        self.noisy_modules: List[Tuple[str, torch.nn.Module]] = None

    def gather_noisy_modules(self):
        """
        Replace regular Conv2d and Linear layer instances with derived approximate layer
        instances that provide additional functionality
        """
        layer_mappings = {
            torch.nn.Conv2d: al.ApproxConv2d,
            torch.nn.Linear: al.ApproxLinear,
        }

        def replace_module(parent_module, base_type, approx_type):
            for name, child_module in parent_module.named_children():
                for child in parent_module.children():
                    replace_module(child, base_type, approx_type)
                if isinstance(child_module, base_type):
                    setattr(parent_module, name, approx_type.from_super(child_module))

        for base_type, approx_type in layer_mappings.items():
            replace_module(self, base_type, approx_type)

        self.noisy_modules = [
            (n, m) for (n, m) in self.named_modules() if isinstance(m, al.ApproxLayer)
        ]
        upgraded_module_names = "\n".join([n for n, _ in self.noisy_modules])
        logger.debug("Upgraded to TorchApprox layers:\n%s", upgraded_module_names)

    @property
    def total_ops(self) -> torch.Tensor:
        """
        Sum of the number of operations for all target layers in the model.
        This is calculated during inference for layers with dynamic input sizes
        like Convolutions.

        Raises:
            ValueError: Opcount has not yet been populated

        Returns:
            Tensor containing a single item with the total number of multiplications
        """
        if self._total_ops is None:
            raise ValueError("Global Opcount is not populated. Run forward pass first.")
        return self._total_ops

    @property
    def mode(self) -> str:
        """
        The current mode of the network. This determines which optimizer
        and number of epochs are selected for optimization runs.
        Can be any of:
        - "baseline": FP32 baseline model
        - "qat": Quantization-aware training
        - "gradient_search": Quantized model with added noise, noise injections
            per layer is optimized together with other network parameters
        - "approx": Approximate Retraining with simulated approximate multipliers
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        if not new_mode in ["baseline", "qat", "approx", "gradient_search"]:
            raise ValueError("Invalide mode")

        def set_mode_params(quant, approx, noise):
            agnapprox.utils.set_all(self, "quantize", quant)
            agnapprox.utils.set_all(self, "approximate", approx)
            agnapprox.utils.set_all(self, "noise", noise)

        self._mode = new_mode
        if self._mode == "baseline":
            set_mode_params(False, False, False)
        if self._mode == "qat":
            set_mode_params(True, False, False)
        if self._mode == "gradient_search":
            set_mode_params(True, False, True)
        if self._mode == "approx":
            set_mode_params(False, True, False)

    def forward(self, features) -> torch.Tensor:
        outputs = self.model(features)
        if self._total_ops is None:
            self._total_ops = sum([m.opcount for _, m in self.noisy_modules])
        return outputs

    def training_step(self, train_batch, _batch_idx) -> torch.Tensor:
        features, labels = train_batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, labels)

        # Add noise loss if we are training noise parameters
        if self._mode == "gradient_search":
            for _, module in self.noisy_modules:
                noise_loss = (module.opcount / self.total_ops) * torch.minimum(
                    torch.abs(module.stdev), torch.tensor(self.sigma_max)
                )
                loss -= self.lmbd * noise_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        accuracies = agnapprox.utils.topk_accuracy(outputs, labels, self.topk)
        for topk, accuracy in zip(self.topk, accuracies):
            self.log(
                "train_acc_top{}".format(topk),
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def validation_step(self, val_batch, _batch_idx) -> torch.Tensor:
        features, labels = val_batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        accuracies = agnapprox.utils.topk_accuracy(outputs, labels, self.topk)
        for topk, accuracy in zip(self.topk, accuracies):
            self.log(
                "val_acc_top{}".format(topk),
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def test_step(self, test_batch, _batch_idx) -> torch.Tensor:
        features, labels = test_batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss, logger=True)
        accuracies = agnapprox.utils.topk_accuracy(outputs, labels, self.topk)
        for topk, accuracy in zip(self.topk, accuracies):
            self.log("test_acc_top{}".format(topk), accuracy, logger=True)
        return loss

    def configure_optimizers(self):
        if self._mode == "baseline":
            return self._baseline_optimizers()
        if self._mode == "qat":
            return self._qat_optimizers()
        if self._mode == "gradient_search":
            return self._gs_optimizers()
        if self._mode == "approx":
            return self._approx_optimizers()
        raise ValueError("Unsupported mode: {}".format(self._mode))

    def _train(
        self,
        datamodule: pl.LightningDataModule,
        run_name: str,
        epochs: Optional[int] = None,
        log_mlflow: bool = False,
        test: bool = False,
        **kwargs
    ):
        """Internal Trainer function. This function is called by the different
        training stage functions.

        Args:
            datamodule: The dataset to train on
            run_name: Run name passed to MLFlow
            epochs: Optional number of epochs to train for.
                If not set, number of epochs defined in the network definition will be used.
                Defaults to None.
            log_mlflow: Log training data to MLFlow. Defaults to False.
            test: Run on test set after training. Defaults to False.
        """
        if epochs is None:
            epochs = self.epochs[self.mode]

        num_gpus = min(self.num_gpus, torch.cuda.device_count())
        device_count = "auto" if num_gpus == 0 else num_gpus

        logger.debug("Training %s - %s on %d GPUs", self.name, run_name, num_gpus)

        trainer = pl.Trainer(
            accelerator="auto", devices=device_count, max_epochs=epochs, **kwargs
        )

        mlflow.pytorch.autolog(log_models=False, disable=not log_mlflow)
        mlflow.set_experiment(experiment_name=self.name)
        with mlflow.start_run(run_name=run_name):
            trainer.fit(self, datamodule)
            if test:
                trainer.test(self, datamodule)

            # Calculate layer assignment results for logging instance
            if self.mode == "gradient_search":
                target_multipliers = EvoApproxLib().prepare(signed=False)
                res = agnapprox.utils.select_multipliers(
                    self, datamodule, target_multipliers, trainer
                )
                agnapprox.utils.deploy_multipliers(self, res, EvoApproxLib())
                agnapprox.utils.dump_results(res, self.lmbd)

    def train_baseline(self, datamodule: pl.LightningDataModule, **kwargs):
        """
        Train an FP32 baseline model

        Args:
            datamodule: Dataset provider
        """
        self.mode = "baseline"
        self._train(datamodule, "Baseline Model", **kwargs)

    def train_quant(self, datamodule: pl.LightningDataModule, **kwargs):
        """
        Train a quantized model using Quantization-Aware training

        Args:
            datamodule: Dataset provider
        """
        self.mode = "qat"
        self._train(datamodule, "QAT Model", **kwargs)

    def train_gradient(
        self,
        datamodule: pl.LightningDataModule,
        lmbd: float = 0.2,
        initial_noise: float = 0.1,
        verbose: bool = False,
        **kwargs
    ):
        """Run Gradient Search algorithm to optimize layer
        robustness parameters

        Args:
            datamodule: Dataset provider
            lmdb: Lambda parameter that controls weighing of
                task loss and noise loss in the overall loss
                function.
                Defaults to 0.2
            initial_noise: The initial value to set for the noise parameter.
                Defaults to 0.1.
            verbose: Print intermediate noise values for each layer after
                each epoch.
                Defaults to False.
        """
        self.mode = "gradient_search"

        # Set initial noise
        with torch.no_grad():
            self.lmbd = lmbd
            for _, module in self.noisy_modules:
                module.stdev = initial_noise

        verbose_callback = [GSVerboseCb()] if verbose else []
        self._train(datamodule, "Gradient Search", callbacks=verbose_callback, **kwargs)

    def train_approx(
        self,
        datamodule: pl.LightningDataModule,
        name_ext: Optional[str] = None,
        **kwargs
    ):
        """Train model with simulated approximate multipliers

        Args:
            datamodule: Dataset provider
            name_ext: Optional extension to add to experiment tracking name.
                Helpful for distinguishing different multiplier configurations
                (i.e. signed/unsigned, uniform/non-uniform, etc.).
                Defaults to None.
        """
        self.mode = "approx"
        name = "Approximate Retraining"
        if name_ext is not None:
            name += name_ext
        self._train(datamodule, name, **kwargs)

    def on_fit_start(self):
        if self.deterministic:
            pl.seed_everything(42, workers=True)

    def _baseline_optimizers(self):
        """
        Baseline Optimizer and Scheduler definition
        """

    def _qat_optimizers(self):
        """
        Quantization-Aware Training Optimizer and Scheduler definition
        """

    def _approx_optimizers(self):
        """
        Approximate Retraining Training Optimizer and Scheduler definition
        """

    def _gs_optimizers(self):
        """
        Gradient Search Optimizer and Scheduler definition
        """


class GSVerboseCb(pl.Callback):
    """
    Callback class implementation that prints out current values
    for sigma_l after each epoch
    """

    def on_train_epoch_end(self, trainer, approxnet):
        # Find longest layer name so that we can align them for printing
        name_len = max(len(name) for name, _ in approxnet.noisy_modules)
        format_str = "Layer: %{}s | sigma_l: %+.3f".format(name_len)
        logger.info("Epoch: %d", trainer.current_epoch)
        for name, module in approxnet.noisy_modules:
            logger.info(format_str, name, module.stdev.item())
