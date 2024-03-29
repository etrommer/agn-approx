"""
Model-level utility functions
"""
import dataclasses
import json
import os
import tempfile
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch

from torchapprox.layers.approx_layer import TracedGeMMInputs

if TYPE_CHECKING:
    from agnapprox.utils.select_multipliers import MatchingInfo


class EnhancedJSONEncoder(json.JSONEncoder):
    # pylint: disable=line-too-long
    """
    Workaround to make dataclasses JSON-serializable
    https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses/51286749#51286749
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def dump_results(result: "MatchingInfo", lmbd: float):
    """
    Write multiplier matching results to MLFlow tracking instance

    Args:
        result: Multiplier Matching Results
        lmbd: Lambda value
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        res_path = os.path.join(temp_dir, "gs_results.json")
        with open(res_path, "w") as handle:
            json.dump(result, handle, indent=4, cls=EnhancedJSONEncoder)
        mlflow.log_artifact(res_path)
        mlflow.log_metric(
            "Relative Energy Consumption", result.relative_energy_consumption
        )
        mlflow.log_param("lambda", lmbd)


def set_all(
    model: Union[pl.LightningDataModule, torch.nn.Module], attr: str, value: Any
):
    """Utility function to set an attribute for all modules in a model

    Args:
        model: The model to set the value on
        attr: Attribute name
        value: Attribute value to set
    """
    for module in model.modules():
        if hasattr(module, attr):
            setattr(module, attr, value)


@dataclasses.dataclass
class IntermediateLayerResults:
    """
    Container that holds the results of running an inference pass
    on sample data with accurate multiplication as well as layer metadata
    For each target layer, we track:
    - `fan_in`: Number of incoming connections
    - `features`: Input activations into the layer for the sample run, squashed
        to a single tensor
    - `outputs`:  Accurate results of the layer for the sample run, squashed
        to a single tensor
    - `weights`: The layer's weights tensor
    """

    fan_in: int
    features: Union[List[np.ndarray], np.ndarray]
    outputs: Union[List[np.ndarray], np.ndarray]
    weights: Optional[np.ndarray] = None

    def finalize(self):
        if self.features.shape[1] != 1:
            self.features = self.features.transpose((0, 2, 1))
            self.weights = self.weights.T
        self.fan_in = self.weights.shape[0]

        return self


# Get approximate op layer inputs, outputs weights and metadata
def get_feature_maps(
    model: pl.LightningModule,
    target_modules: List[Tuple[str, torch.nn.Module]],
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
) -> Dict[str, IntermediateLayerResults]:
    """
    Capture intermediate feature maps of a model's layer
    by attaching hooks and running sample data

    Args:
        model: The neural network model to gather IFMs from
        target_modules: List of modules in the network for which IFMs should be gathered
        trainer: A PyTorch Lightning Trainer instance that is used to run the inference
        datamodule: PyTorch Lightning DataModule instance that is used to generate input sample data

    Returns:
        Dictionary with Input IFM, Output IFM, Weights Tensor and Fan-In for each target layer
    """
    results = {}

    # TODO: Set LUTs to None to force accurate calculation
    prev_mode = model.mode
    model.mode = "approx"

    for _, m in target_modules:
        m.traced_inputs = TracedGeMMInputs(None, None)

    # Run validation to populate
    trainer.validate(model, datamodule.sample_dataloader(), verbose=False)

    for n, m in target_modules:
        results[n] = IntermediateLayerResults(
            0,
            m.traced_inputs.features.numpy(),
            None,
            m.traced_inputs.weights.numpy(),
        ).finalize()
        m.traced_inputs = None

    model.mode = prev_mode

    return results


# Taken from: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    # pylint: disable=line-too-long
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    Args:
        output: output is the prediction of the model e.g. scores, logits, raw y_pred
            before normalization or getting classes
        target: target is the truth
        topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
            e.g. in top 2 it means you get a +1 if your models's top 2 predictions
            are in the right label.
            So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
            but if it were either cat or dog you'd accumulate +1 for that example.
    Returns:
        list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes]
        # (i.e. the number of most likely probabilities we will use)
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores
        # just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()
        # [B, maxk] -> [maxk, B]
        # Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is
        # in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth.
        # If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match
        # (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give
        # credit if any matches the ground truth
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to
            # get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc.item())
        # list of topk accuracies for entire batch [topk1, topk2, ... etc]
        return list_topk_accs
