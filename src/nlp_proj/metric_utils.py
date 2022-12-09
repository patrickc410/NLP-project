import torch
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


def accuracy_torch(
    preds: Tensor,
    labels: Tensor,
    num_classes: int,
    average: str = "micro",
    device: torch.device = None,
) -> float:
    """Accuracy for tensors

    Args:
        preds (torch.Tensor): predictions as a (N,) tensor
        labels (torch.Tensor): labels as a (N,) tensor

    Returns:
        float: accuracy score
    """
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average=average)
    if device:
        acc_metric = acc_metric.to(device)
    acc_tensor = acc_metric(preds, labels)
    return acc_tensor.item()


def f1_torch(
    preds: Tensor,
    labels: Tensor,
    num_classes: int,
    average: str = "macro",
    device: torch.device = None,
) -> float:
    """F1 Score for tensors

    Args:
        preds (torch.Tensor): predictions as a (N,) tensor
        labels (torch.Tensor): labels as a (N,) tensor
        num_classes (int): number of classes in the classification
        average (str, optional): method for aggregating over labels. Defaults to "macro".
            See for details: https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html?highlight=F1

    Returns:
        float: F1 score
    """
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average)
    if device:
        f1_metric = f1_metric.to(device)
    f1_tensor = f1_metric(preds, labels)
    return f1_tensor.item()
