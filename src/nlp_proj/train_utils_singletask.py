import wandb
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import Tuple, Dict
from types import SimpleNamespace
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score
from nlp_proj.metric_utils import f1_torch, accuracy_torch
import logging


def train_batch_singletask(
    batch_x: transformers.tokenization_utils_base.BatchEncoding,
    batch_y: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: SimpleNamespace,
) -> Tuple[float, Tensor]:
    """Encapsulate the logic of a single batched training optimization step
    NOTE: assumes that model, batch_x, batch_y are on the same device already

    Args:
        batch_x (transformers.tokenization_utils_base.BatchEncoding): _description_
        batch_y (Tensor): _description_
        model (nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        criterion (nn.Module): _description_
        config

    Returns:
        Tuple[float, Tensor]: loss, model outputs
    """

    # Forward pass
    input_ids = batch_x.input_ids
    attention_mask = batch_x.attention_mask
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Loss calculation
    if config.task == "regression":
        batch_y = batch_y.float()
    loss = criterion(outputs, batch_y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    if hasattr(config, "clip_grad"):
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

    # Step with optimizer
    optimizer.step()

    return loss, outputs


def train_model_singletask(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: SimpleNamespace,
):
    """Single-task learning: Full training loop"""
    # Device
    device = config.device
    model = model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    last_val_evaluation = None
    for epoch in tqdm(range(config.max_epochs)):

        train_evals = {"loss": []}
        if config.task == "classification":
            train_evals["acc"] = []
            train_evals["f1"] = []
        elif config.task == "regression":
            train_evals["mae"] = []
            train_evals["mse"] = []
            train_evals["r2"] = []

        for _, (batch_x, batch_y) in enumerate(train_loader):
            # Push batch_x, batch_y to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Train
            # fmt: off
            loss, outputs = train_batch_singletask(batch_x, batch_y, model, optimizer, criterion, config)
            train_evals["loss"].append(loss)
            # fmt: on

            # Predictions and Train Eval
            # fmt: off
            if config.task == "classification":
                preds = torch.argmax(outputs, dim=-1)
                train_evals["acc"].append(accuracy_torch(preds, batch_y, num_classes=config.num_classes, device=device))
                train_evals["f1"].append(f1_torch(preds, batch_y, num_classes=config.num_classes, device=device))
            elif config.task == "regression":
                preds = outputs.squeeze()
                train_evals["mae"].append(mean_absolute_error(preds, batch_y))
                train_evals["mse"].append(mean_squared_error(preds, batch_y))
                train_evals["r2"].append(r2_score(preds, batch_y))
            # fmt: on

            # Increment
            example_ct += len(batch_y)
            batch_ct += 1

            # Report metrics every batch
            if ((batch_ct + 1) % config.logging_freq) == 0:
                # fmt: off
                log_dict = {"epoch": epoch, "batch_ct": batch_ct}
                for metric_name, values in train_evals.items():
                    log_dict[f"train_batch_{metric_name}"] = values[-1]
                    logging.info(f"Epoch {epoch}, Training   {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {values[-1]:.3f}")
                wandb.log(log_dict, step=example_ct)
                # fmt: on

        # Average training evaluation over epoch
        # fmt: off
        train_mean_evals = {
            metric_name: torch.mean(torch.tensor(value))
            for metric_name, value in train_evals.items()
        }
        log_dict = {"epoch": epoch, "batch_ct": batch_ct}
        for metric_name, value in train_mean_evals.items():
            log_dict[f"train_{metric_name}"] = value
            logging.info(f"Epoch {epoch}, Training   {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {value:.3f}")
        wandb.log(log_dict, step=example_ct)
        # fmt: on

        # Validation evaluation
        # fmt: off
        val_evaluation, _ = test_model_singletask(model, val_loader, config)
        log_dict = {"epoch": epoch, "batch_ct": batch_ct}
        for metric_name, value in val_evaluation.items():
            log_dict[f"val_{metric_name}"] = value
            logging.info(f"Epoch {epoch}, Validation {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {value:.3f}")
        wandb.log(log_dict, step=example_ct)
        # fmt: on

        # Early stopping check
        if config.early_stopping is True and last_val_evaluation is not None:
            met = config.early_stopping_metric
            if val_evaluation[met] < last_val_evaluation[met]:
                logging.info(f"Epoch {epoch}, Early stopping")
                break
        last_val_evaluation = {
            metric_name: value for metric_name, value in val_evaluation.items()
        }

    return model


def test_model_singletask(
    model: nn.Module,
    test_loader: DataLoader,
    config: SimpleNamespace,
) -> Tuple[Dict[str, float], int]:
    """Test model performance over data in test_loader

    Args:
        model (nn.Module): _description_
        test_loader (DataLoader): _description_
        num_classes (int): _description_
        device (torch.device): _description_
        config

    Returns:
        Tuple[Dict[str, float], int]:
            dictionary with evaluation metrics, and number of samples tested on
    """
    # Device
    device = config.device
    model = model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    # Run the model on some test examples
    with torch.no_grad():
        total = 0
        for batch_x, batch_y in test_loader:
            # Push batch_x to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            input_ids = batch_x.input_ids
            attention_mask = batch_x.attention_mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Predictions
            if config.task == "classification":
                batch_preds = torch.argmax(outputs, dim=-1)
            elif config.task == "regression":
                batch_preds = outputs.squeeze()

            all_preds.append(batch_preds)
            all_labels.append(batch_y)
            total += batch_y.size(0)

    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)

    # Evaluation
    evaluation = {}
    if config.task == "classification":
        evaluation["acc"] = accuracy_torch(all_preds, all_labels, config.num_classes)
        evaluation["f1"] = f1_torch(all_preds, all_labels, config.num_classes)
    elif config.task == "regression":
        evaluation["mae"] = mean_absolute_error(all_preds, all_labels)
        evaluation["mse"] = mean_squared_error(all_preds, all_labels)
        evaluation["r2"] = r2_score(all_preds, all_labels)

    return evaluation, total
