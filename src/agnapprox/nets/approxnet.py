"""
Approximate Neural Network boilerplate implementation
"""
# pylint: disable=arguments-differ
import logging
from typing import List, Optional, Tuple

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchapprox.layers as tal
from torchapprox.utils.conversion import get_approx_modules, wrap_quantizable
from torch.ao.quantization import prepare_qat, QConfig

import agnapprox.utils

# from agnapprox.libs.evoapprox import EvoApprox

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class ApproxNet(pl.LightningModule):
    """
    Base Class that provideds common functionality for approximate neural network training
    """

    def __init__(
        self,
        deterministic: bool = False,
    ):
        super().__init__()

        self._mode: str = "baseline"
        self._total_ops: Optional[int] = None
        self.deterministic: bool = deterministic
        self.approx_modules: List[Tuple[str, torch.nn.Module]] = []
        self.qconfig: Optional[QConfig] = None

    def convert(self):
        """
        Replace regular Conv2d and Linear layer instances with derived approximate layer
        instances that provide additional functionality
        """
        self.model = wrap_quantizable(self.model, self.qconfig)
        prepare_qat(
            self.model, mapping=tal.approx_wrapper.layer_mapping_dict(), inplace=True
        )

        self.approx_modules = get_approx_modules(self)
        upgraded_module_names = "\n".join([n for n, _ in self.approx_modules])
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
        if new_mode not in ["baseline", "qat", "approx"]:
            raise ValueError("Invalide mode")

        self._mode = new_mode
        if self._mode == "qat":
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.QUANTIZED
        if self._mode == "approx":
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.APPROXIMATE

    def forward(self, features) -> torch.Tensor:
        outputs = self.model(features)
        return outputs

    def training_step(self, train_batch, _batch_idx) -> torch.Tensor:
        features, labels = train_batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, labels)

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
        if self._mode == "approx":
            return self._approx_optimizers()
        raise ValueError("Unsupported mode: {}".format(self._mode))

    def _train(
        self,
        datamodule: pl.LightningDataModule,
        run_name: str,
        epochs: Optional[int] = None,
        log_mlflow: bool = True,
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
                if self.mode == "noise" or self.mode == "approx":
                    for _, m in self.approx_modules:
                        m.inference_mode = tal.InferenceMode.APPROXIMATE
                        m.htp_model = None
                trainer.test(self, datamodule)

    def on_train_epoch_start(self) -> None:
        if self.mode == "qat" and self.current_epoch == 2:
            self.model.apply(torch.ao.quantization.disable_observer)
            self.model.apply(torch.ao.quantization.enable_fake_quant)

    def train_baseline(self, datamodule: pl.LightningDataModule, **kwargs):
        """
        Train an FP32 baseline model

        Args:
            datamodule: Dataset provider
        """
        self.mode = "baseline"
        self._train(datamodule, "Baseline Model", **kwargs)

        self.convert()
        self.model.apply(torch.ao.quantization.enable_observer)
        self.model.apply(torch.ao.quantization.disable_fake_quant)
        self.mode = "qat"
        self._train(datamodule, "Quantized Model", **kwargs)

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
