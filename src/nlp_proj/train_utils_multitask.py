import wandb
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import Tuple, Dict, List
from types import SimpleNamespace
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score
from nlp_proj.metric_utils import f1_torch, accuracy_torch
import logging


def train_batch_multitask(
    batch_x: transformers.tokenization_utils_base.BatchEncoding,
    batch_y: Dict[str, Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: List[nn.Module],
    config: SimpleNamespace,
) -> Tuple[float, List[float], List[Tensor]]:
    """MULTITASK. Encapsulate the logic of a single batched training optimization step
    NOTE: assumes that model, batch_x, batch_y are on the same device already

    Args:
        batch_x (transformers.tokenization_utils_base.BatchEncoding): tokenized tensors to input
        batch_y (Dict[str, Tensor]): dictionary with keys as label column names and values as tensors
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        criterion (List[nn.Module]): list of loss criterion
        config

    Returns:
        Tuple[float, List[float], List[Tensor]]:
            averaged loss across all heads,
            list of losses for all heads
            list of model output tensors for all heads
    """
    model.train()

    # Forward pass
    input_ids = batch_x.input_ids
    attention_mask = batch_x.attention_mask
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Loss calculation
    losses = []
    for output, c, crit, task in zip(
        outputs, config.label_cols, criterion, config.tasks
    ):
        batch_y_c = batch_y[c]
        if task == "regression":
            batch_y_c = batch_y_c.float()
        loss = crit(output, batch_y_c)
        losses.append(loss)
    loss = torch.concat([t.unsqueeze(0) for t in losses])
    loss = torch.mean(loss)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    if hasattr(config, "clip_grad"):
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

    # Step with optimizer
    optimizer.step()

    return loss, losses, outputs


def train_model_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: List[nn.Module],
    optimizer: torch.optim.Optimizer,
    config: SimpleNamespace,
):
    """Full training loop for multitask"""
    # Device
    device = config.device
    model = model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.max_epochs)):

        # Setup training evaluation dictionaries
        train_evals = {c: {"loss": []} for c in config.label_cols}
        for c, task in zip(config.label_cols, config.tasks):
            if task == "classification":
                train_evals[c]["acc"] = []
                train_evals[c]["f1"] = []
            elif task == "regression":
                train_evals[c]["mae"] = []
                train_evals[c]["mse"] = []
                train_evals[c]["r2"] = []

        for _, (batch_x, batch_y) in enumerate(train_loader):
            # Push batch_x to device
            batch_x = batch_x.to(device)

            # Push batch_y to device multitask
            for label_col, labels_batch in batch_y.items():
                batch_y[label_col] = labels_batch.to(device)

            # Train
            loss, losses, outputs = train_batch_multitask(
                batch_x, batch_y, model, optimizer, criterion, config
            )
            for c, c_loss in zip(config.label_cols, losses):
                train_evals[c]["loss"].append(c_loss)

            # Predictions and Evaluation; skip for batch size of 1
            # fmt: off
            if len(batch_y[config.label_cols[0]]) > 1:
                for output, c, task, classes in zip(outputs, config.label_cols, config.tasks, config.num_classes_list):
                    batch_y_c = batch_y[c]
                    if task == "classification":
                        preds = torch.argmax(output, dim=-1)
                        train_evals[c]["acc"].append(accuracy_torch(preds, batch_y_c, num_classes=classes, device=device))
                        train_evals[c]["f1"].append(f1_torch(preds, batch_y_c, num_classes=classes, device=device))
                    elif task == "regression":
                        preds = output.squeeze()
                        train_evals[c]["mae"].append(mean_absolute_error(preds, batch_y_c))
                        train_evals[c]["mse"].append(mean_squared_error(preds, batch_y_c))
                        train_evals[c]["r2"].append(r2_score(preds, batch_y_c))
            # fmt: on

            # Increment
            c = config.label_cols[0]
            example_ct += len(batch_y[c])
            batch_ct += 1

            # Report metrics every batch
            if ((batch_ct + 1) % config.logging_freq) == 0:
                # fmt: off
                log_dict = {"epoch": epoch, "batch_ct": batch_ct}
                for c, metric_dict in train_evals.items():
                    for metric_name, values in metric_dict.items():
                        log_dict[f"train_batch_{c}_{metric_name}"] = values[-1]
                        logging.info(f"Epoch {epoch}, Training   {c.ljust(13)} {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {values[-1]:.3f}")
                wandb.log(log_dict, step=example_ct)
                # fmt: on

        train_mean_evals = {
            c: {
                metric_name: torch.mean(torch.tensor(value))
                for metric_name, value in metric_dict.items()
            }
            for c, metric_dict in train_evals.items()
        }
        log_dict = {"epoch": epoch, "batch_ct": batch_ct}
        for c, metric_dict in train_mean_evals.items():
            for metric_name, value in metric_dict.items():
                log_dict[f"train_{c}_{metric_name}"] = value
                logging.info(
                    f"Epoch {epoch}, Training   {c.ljust(13)} {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {value:.3f}"
                )
        wandb.log(log_dict, step=example_ct)

        # fmt: off
        val_evaluation, _ = test_model_multitask(model, val_loader, config)
        log_dict = {"epoch": epoch, "batch_ct": batch_ct}
        for c, metric_dict in val_evaluation.items():
            for metric_name, value in metric_dict.items():
                log_dict[f"val_{c}_{metric_name}"] = value
                logging.info(f"Epoch {epoch}, Validation {c.ljust(12)} {metric_name.ljust(6)} after {str(batch_ct).zfill(5)} batches: {value:.3f}")
        # fmt: on

    return model


def test_model_multitask(
    model: nn.Module,
    test_loader: DataLoader,
    config: SimpleNamespace,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Test model performance over data in test_loader

    Args:
        model (nn.Module): _description_
        test_loader (DataLoader): _description_
        num_classes (int): _description_
        device (torch.device): _description_
        config

    Returns:
        Tuple[Dict[str, Dict[str, float]], int]:
            dictionary with evaluation metrics, and
            number of samples tested on
    """
    # Device
    device = config.device
    model = model.to(device)

    model.eval()

    all_preds = {c: [] for c in config.label_cols}
    all_labels = {c: [] for c in config.label_cols}

    # Run the model on some test examples
    with torch.no_grad():
        total = 0
        for batch_x, batch_y in test_loader:
            # Push batch_x to device
            batch_x = batch_x.to(device)

            # Push batch_y to device multitask
            for label_col, labels_batch in batch_y.items():
                batch_y[label_col] = labels_batch.to(device)

            # Forward
            input_ids = batch_x.input_ids
            attention_mask = batch_x.attention_mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Predictions
            for output, c, task in zip(outputs, config.label_cols, config.tasks):
                batch_y_c = batch_y[c]
                if task == "classification":
                    batch_preds = torch.argmax(output, dim=-1)
                elif task == "regression":
                    batch_preds = output.squeeze()

                all_preds[c].append(batch_preds)
                all_labels[c].append(batch_y_c)

            c = config.label_cols[0]
            total += batch_y[c].size(0)

    all_preds = {c: torch.concat(preds).to(device) for c, preds in all_preds.items()}
    all_labels = {
        c: torch.concat(labels).to(device) for c, labels in all_labels.items()
    }

    # Evaluation
    evaluation = {c: {} for c in config.label_cols}
    for c, task, classes in zip(
        config.label_cols, config.tasks, config.num_classes_list
    ):
        preds = all_preds[c]
        labels = all_labels[c]
        if task == "classification":
            evaluation[c]["acc"] = accuracy_torch(
                preds, labels, num_classes=classes, device=device
            )
            evaluation[c]["f1"] = f1_torch(
                preds, labels, num_classes=classes, device=device
            )
        elif task == "regression":
            evaluation[c]["mae"] = mean_absolute_error(preds, labels)
            evaluation[c]["mse"] = mean_squared_error(preds, labels)
            evaluation[c]["r2"] = r2_score(preds, labels)

    return evaluation, total
