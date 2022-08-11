"""
Model-level utility functions
"""
from operator import attrgetter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch


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


# Get approximate op layer inputs, outputs weights and metadata
def get_feature_maps(
    model: pl.LightningModule,
    target_modules: List[Tuple[str, torch.nn.Module]],
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
) -> Dict[str, Dict[str, Union[np.array, float]]]:
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

    # Create hook function for each layer
    def get_hook(name):
        module_getter = attrgetter(name)
        results[name] = {
            "input": [],
            "output": [],
            "weights": None,
            "fan_in": module_getter(model).fan_in,
        }

        def hook(_module, module_in, module_out):
            if results[name]["weights"] is None:
                results[name]["weights"] = (
                    module_in[1].cpu().detach().numpy().astype(np.float32)
                )
            results[name]["input"].append(
                module_in[0].cpu().detach().numpy().astype(np.float32)
            )
            results[name]["output"].append(
                module_out.cpu().detach().numpy().astype(np.float32)
            )

        return hook

    # Set hooks
    handles = [
        target_module.approx_op.register_forward_hook(get_hook(name))
        for name, target_module in target_modules
    ]

    # TODO: Set LUTs to None to force accurate calculation
    set_all(model, "approximate", True)

    # Run validation to populate
    trainer.validate(model, datamodule.sample_dataloader(), verbose=False)

    # Squash batches to a single array
    for layer, result in results.items():
        results[layer]["input"] = np.concatenate(result["input"])
        results[layer]["output"] = np.concatenate(result["output"])

    # Clean up
    _ = [h.remove() for h in handles]
    set_all(model, "approximate", False)

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
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1
            ).float()  # [k, B] -> [kB]
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
