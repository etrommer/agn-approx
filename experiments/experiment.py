from typing import Optional
from agnapprox.datamodules.approx_datamodule import ApproxDataModule
from agnapprox.nets.approxnet import ApproxNet
import os
import pytorch_lightning as pl
import torch
import logging
import pathlib
import copy
import torch.ao.quantization as quant

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ApproxExperiment:
    def __init__(
        self,
        model: ApproxNet,
        datamodule: ApproxDataModule,
        name: str,
        model_dir: str = "./models",
        test: bool = False,
    ) -> None:
        self.model_fp32: ApproxNet = model
        self.model_quant: Optional[ApproxNet] = None
        self.datamodule: ApproxDataModule = datamodule
        self.name = name
        self.model_dir = model_dir
        self.test = test

        self.datamodule.prepare_data()
        self.datamodule.setup()

    @property
    def baseline_model(self) -> ApproxNet:
        baseline_path = os.path.join(self.model_dir, f"{self.name.lower()}_baseline.pt")
        if not os.path.exists(baseline_path):
            logger.debug(f"No baseline model found in {baseline_path}. Training a new one.")
            pl.seed_everything(42)
            self.model_fp32.train_baseline_fp32(self.datamodule, test=self.test)
            if not os.path.exists(self.model_dir):
                logger.debug(f"Model directory {self.model_dir} not found. Creating.")
                pathlib.Path(self.model_dir).mkdir()
            torch.save(self.model_fp32.state_dict(), baseline_path)
        self.model_fp32.load_state_dict(torch.load(baseline_path))
        if torch.cuda.is_available():
            self.model_fp32.to("cuda")
        return copy.deepcopy(self.model_fp32)

    def quantized_model(self, qconfig: quant.QConfig, qtype: str) -> ApproxNet:
        quant_model = self.baseline_model
        quant_path = os.path.join(self.model_dir, f"{self.name.lower()}_{qtype.lower()}.pt")
        quant_model.qconfig = qconfig

        if not os.path.exists(quant_path):
            logger.debug(f"No quantized model for mode {qtype} found in {quant_path}. Training a new one.")
            pl.seed_everything(42)
            quant_model.train_baseline_quant(self.datamodule, test=self.test, mlf_params={"qtype": qtype})
            torch.save(quant_model.state_dict(), quant_path)
        else:
            quant_model.convert()

        quant_model.load_state_dict(torch.load(quant_path))
        if torch.cuda.is_available():
            quant_model.to("cuda")
        return quant_model
