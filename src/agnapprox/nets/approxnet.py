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
from torchapprox.utils.conversion import (
    get_approx_modules,
    wrap_quantizable,
    convert_batchnorms,
)
import numpy.typing as npt
from torch.ao.quantization import prepare_qat, QConfig
import torch.nn.utils.prune as prune

import agnapprox.utils

# from agnapprox.libs.evoapprox import EvoApprox

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class ApproxNet(pl.LightningModule):
    """
    Base Class that provideds common functionality for approximate neural network training
    """

    def __init__(self, deterministic: bool = True):
        super().__init__()

        self._mode: str = "baseline"
        self._total_ops: Optional[int] = None
        self.deterministic: bool = deterministic
        self.approx_modules: List[Tuple[str, torch.nn.Module]] = []
        self.qconfig: Optional[QConfig] = None
        self.name: str = ""
        self.multi_retraining_size: Optional[int] = None

    def init_shadow_luts(self, luts: npt.NDArray):
        assert luts.shape[0] == len(
            self.approx_modules
        ), "First array dimension must correspond to layers"
        assert luts.shape[-1] == luts.shape[-2] == 256, "LUTs must be last dimension"
        for (n, m), layer_luts in zip(self.approx_modules, luts):
            if hasattr(m, "init_shadow_luts"):
                logger.debug(
                    f"Setting {n} to multi-training with {len(layer_luts)} LUTs"
                )
                m.init_shadow_luts(layer_luts)
        convert_batchnorms(self, len(luts[0]))
        self.automatic_optimization = False
        self.multi_retraining_size = len(luts[0])

    def convert(self):
        """
        Replace regular Conv2d and Linear layer instances with derived approximate layer
        instances that provide additional functionality
        """
        if self.qconfig is None:
            raise ValueError(
                "Converting to quantization without attaching a valid QConfig. Set model.qconfig = torch.ao.quantization.Qconfig(...) before conversion."
            )
        self.model = wrap_quantizable(self.model, qconfig=self.qconfig)
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
    def mul_idx(self) -> Optional[int]:
        for m in self.modules():
            if hasattr(m, "mul_idx"):
                return m.mul_idx
        return None

    @mul_idx.setter
    def mul_idx(self, new_idx: int):
        for m in self.modules():
            if m == self:
                continue
            if hasattr(m, "mul_idx"):
                m.mul_idx = new_idx

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
        if new_mode not in ["noise", "qat", "approx", "prune"]:
            raise ValueError("Invalide mode")

        self._mode = new_mode
        if self._mode == "qat":
            self.model.apply(torch.ao.quantization.enable_observer)
            # Fake-quant is enabled after two warm-up epochs to calibrate observers
            self.model.apply(torch.ao.quantization.disable_fake_quant)
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.QUANTIZED
        if self._mode == "approx":
            self.model.apply(torch.ao.quantization.enable_observer)
            self.model.apply(torch.ao.quantization.enable_fake_quant)
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.APPROXIMATE
        if self.mode == "noise":
            self.model.apply(torch.ao.quantization.enable_observer)
            self.model.apply(torch.ao.quantization.enable_fake_quant)
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.NOISE

    def forward(self, features) -> torch.Tensor:
        outputs = self.model(features)
        if self._total_ops is None and self.mode != "baseline":
            self._total_ops = sum([m.opcount for _, m in self.approx_modules])
        return outputs

    def _log_loss(self, loss, logger_item):
        self.log(
            logger_item,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def _log_accuracies(self, outputs, labels, logger_item):
        accuracies = agnapprox.utils.topk_accuracy(outputs, labels, self.topk)
        for topk, accuracy in zip(self.topk, accuracies):
            self.log(
                f"{logger_item}_top{topk}",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    def _automatic_training_step(self, train_batch, _batch_idx) -> torch.Tensor:
        features, labels = train_batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, labels)
        if self.mode == "noise":
            for _, mod in self.approx_modules:
                noise_loss = (mod.opcount / self.total_ops) * torch.minimum(
                    torch.abs(mod.stdev), torch.tensor(self.sigma_max)
                )
                loss -= torch.tensor(self.lmbd) * noise_loss

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

    def _multi_training_step(self, train_batch, _batch_idx) -> None:
        from torchapprox.layers.multi_batchnorm import MultiBatchNorm

        for i in range(self.multi_retraining_size):
            self.mul_idx = i
            opt = self.optimizers()
            opt.zero_grad()
            features, labels = train_batch
            outputs = self(features)
            loss = F.cross_entropy(outputs, labels)
            self.manual_backward(loss)
            for n, p in self.named_parameters():
                if ("bias" in n) or ("fwd_norm" in n):
                    continue
                if p.grad is None:
                    continue
                p.grad /= self.multi_retraining_size
            b = False
            for n, m in self.named_modules():
                if isinstance(m, MultiBatchNorm):
                    print(m._mul_idx, m.fwd_norm.weight)
                    b = True
                if b:
                    break
                    # for sn in m._shadow_norms:
                    #     print(n, i, sn.weight.grad.cpu().flatten()[:10])
            opt.step()
            self._log_loss(loss, f"train_loss{i}")
            self._log_accuracies(outputs, labels, f"train_acc{i}")
        return

    def training_step(self, train_batch, _batch_idx):
        if self.automatic_optimization:
            return self._automatic_training_step(train_batch, _batch_idx)
        else:
            return self._multi_training_step(train_batch, _batch_idx)

    def validation_step(self, val_batch, _batch_idx) -> torch.Tensor:
        features, labels = val_batch
        if not self.automatic_optimization:
            for i in range(self.multi_retraining_size):
                self.mul_idx = i
                outputs = self(features)
                loss = F.cross_entropy(outputs, labels)
                self._log_loss(loss, f"val_loss{i}")
                self._log_accuracies(outputs, labels, f"val_acc{i}")
        else:
            outputs = self(features)
            loss = F.cross_entropy(outputs, labels)
            self._log_loss(loss, "val_loss")
            self._log_accuracies(outputs, labels, "val_acc")
        return loss

    def test_step(self, test_batch, _batch_idx) -> torch.Tensor:
        if not self.automatic_optimization:
            for i in range(self.multi_retraining_size):
                self.mul_idx = i
                features, labels = test_batch
                outputs = self(features)
                loss = F.cross_entropy(outputs, labels)
                self._log_loss(loss, f"test_loss{i}")
                self._log_accuracies(outputs, labels, f"test_acc{i}")
        else:
            features, labels = test_batch
            outputs = self(features)
            loss = F.cross_entropy(outputs, labels)
            self._log_loss(loss, "test_loss")
            self._log_accuracies(outputs, labels, "test_acc")
        return loss

    def configure_optimizers(self):
        if self._mode == "baseline":
            return self._baseline_optimizers()
        if self._mode == "noise":
            return self._qat_optimizers()
        if self._mode == "qat":
            return self._qat_optimizers()
        if self._mode == "approx":
            return self._approx_optimizers()
        if self._mode == "prune":
            return self._prune_optimizers()
        raise ValueError("Unsupported mode: {}".format(self._mode))

    def _train(
        self,
        datamodule: pl.LightningDataModule,
        run_name: str,
        epochs: Optional[int] = None,
        log_mlflow: bool = True,
        test: bool = False,
        **kwargs,
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

        mlf_extra_params = kwargs.pop("mlf_params", {})
        mlf_artifacts = kwargs.pop("mlf_artifacts", [])

        trainer = pl.Trainer(
            accelerator="auto", devices=device_count, max_epochs=epochs, **kwargs
        )

        mlflow.pytorch.autolog(log_models=False, disable=not log_mlflow)
        mlflow.set_experiment(experiment_name=self.name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(mlf_extra_params)
            for artifact in mlf_artifacts:
                mlflow.log_artifact(artifact)
            trainer.fit(self, datamodule)
            if test:
                trainer.test(self, datamodule)

    def on_train_epoch_start(self) -> None:
        if self.mode == "qat" and self.current_epoch == 2:
            self.model.apply(torch.ao.quantization.enable_observer)
            self.model.apply(torch.ao.quantization.enable_fake_quant)
        if self.mode == "approx" and self.current_epoch == 2:
            pass
            # self.model.apply(torch.ao.quantization.disable_observer)
        if self.mode == "prune":
            for n, m in self.approx_modules:
                target_sparsity = getattr(m, "target_sparsity", None)
                if target_sparsity and self.current_epoch <= self.pruning_epochs:
                    prev_sparsity = 1 - torch.count_nonzero(m.weight) / m.weight.numel()
                    prune_amount = 1 - (
                        (1 - m.target_sparsity) ** (1 / self.pruning_epochs)
                    )
                    prune.l1_unstructured(m, "weight", prune_amount)
                    curr_sparsity = 1 - torch.count_nonzero(m.weight) / m.weight.numel()
                    logger.error(
                        f"Pruned {n} from {prev_sparsity:5.4f} to {curr_sparsity:5.4f}"
                    )

    def on_train_epoch_end(self) -> None:
        if self.mode == "noise":
            for n, m in self.approx_modules:
                logger.error(
                    f"{n} : sigma = {abs(m.stdev.item()):5.4f} @ {m.opcount:9} ops"
                )

    def on_test_start(self) -> None:
        self.model.apply(torch.ao.quantization.disable_observer)
        if self.mode == "approx":
            for _, m in self.approx_modules:
                m.inference_mode = tal.InferenceMode.APPROXIMATE
                m.htp_model = None

    def on_validation_start(self) -> None:
        pass
        # self.model.apply(torch.ao.quantization.disable_observer)

    def on_validation_end(self) -> None:
        pass
        # Slight hack:
        # Setting the current mode sets the training configuratin for the current mode
        # self.mode = self.mode

    def train_baseline_fp32(self, datamodule: pl.LightningDataModule, **kwargs):
        """
        Train an FP32 baseline model

        Args:
            datamodule: Dataset provider
        """
        self._train(datamodule, "Baseline Model", **kwargs)

    def train_prune(self, datamodule: pl.LightningDataModule, **kwargs):
        self.mode = "prune"
        self._train(datamodule, "Noise Search", **kwargs)

    def train_noise(self, datamodule: pl.LightningDataModule, **kwargs):
        self.convert()
        self.mode = "noise"
        self._train(datamodule, "Noise Search", **kwargs)

    def train_baseline_quant(self, datamodule: pl.LightningDataModule, **kwargs):
        self.convert()
        self.mode = "qat"
        self._train(datamodule, "Quantized Model", **kwargs)

    def train_approx(
        self,
        datamodule: pl.LightningDataModule,
        name_ext: Optional[str] = None,
        **kwargs,
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

    def _noise_optimizers(self):
        """
        Gradient-base Robustness Search Optimizer and Scheduler definition
        """

    def _qat_optimizers(self):
        """
        Quantization-Aware Training Optimizer and Scheduler definition
        """

    def _approx_optimizers(self):
        """
        Approximate Retraining Training Optimizer and Scheduler definition
        """

    def _prune_optimizers(self):
        """
        Pruning Optimizer and Scheduler definition
        """
