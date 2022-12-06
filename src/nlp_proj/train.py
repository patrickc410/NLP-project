import wandb
import argparse
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import transformers
from typing import Tuple, Dict, List, Union
from types import SimpleNamespace
from pprint import pformat

# from dotenv import load_dotenv
import logging

from nlp_proj.data_loader import WikiManualActiveVoiceDataset
from nlp_proj.model_optim_utils import make_model, make_optimizer, make_criterion
from nlp_proj.config_utils import load_config
from nlp_proj.dataset_utils import make_dataloader, make_tokenizer, make_datasets

# load_dotenv()
logging.getLogger().setLevel(logging.INFO)


def accuracy_torch(
    preds: Tensor, labels: Tensor, num_classes: int, average: str = "micro"
) -> float:
    """Accuracy for tensors

    Args:
        preds (torch.Tensor): predictions as a (N,) tensor
        labels (torch.Tensor): labels as a (N,) tensor

    Returns:
        float: accuracy score
    """
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average=average)
    acc_tensor = acc_metric(preds, labels)
    return acc_tensor.item()


def f1_torch(
    preds: Tensor, labels: Tensor, num_classes: int, average: str = "macro"
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
    f1_tensor = f1_metric(preds, labels)
    return f1_tensor.item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: SimpleNamespace,
):
    """Full training loop"""
    # Device
    device = config.device
    model = model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.max_epochs)):

        train_accs = []
        train_f1s = []

        for _, (batch_x, batch_y) in enumerate(train_loader):
            # Push batch_x to device
            batch_x = batch_x.to(device)

            # Push batch_y to device multitask
            if config.multitask is True:
                for label_col, labels_batch in batch_y.items():
                    batch_y[label_col] = labels_batch.to(device)

            # Push batch_y to device single task (and extract from batch_y dictionary)
            else:
                batch_y = batch_y[config.label_col].to(device)

            # Train
            loss, outputs = train_batch(
                batch_x, batch_y, model, optimizer, criterion, config
            )

            # Predictions
            preds = torch.argmax(outputs, dim=-1)

            # Train Eval
            acc = accuracy_torch(preds, batch_y, num_classes=config.num_classes)
            f1 = f1_torch(preds, batch_y, num_classes=config.num_classes)
            train_accs.append(acc)
            train_f1s.append(f1)

            # Increment
            example_ct += len(batch_y)
            batch_ct += 1

            # Report metrics every batch
            if ((batch_ct + 1) % 5) == 0:
                # fmt: off
                wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
                logging.info(f"Epoch {epoch}, Training Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
                # fmt: on

        # fmt: off
        train_acc = torch.mean(torch.tensor(train_accs))
        train_f1 = torch.mean(torch.tensor(train_f1s))
        wandb.log({"epoch": epoch, "train_acc": train_acc, "train_f1": train_f1}, step=example_ct)
        logging.info(f"Epoch {epoch}, Training Acc   after {str(example_ct).zfill(5)} examples: {train_acc:.3f}")
        logging.info(f"Epoch {epoch}, Training F1    after {str(example_ct).zfill(5)} examples: {train_f1:.3f}")
        # fmt: on

        # fmt: off
        val_acc, val_f1, _ = test_model(model, val_loader, config.num_classes, config.device, config)
        wandb.log({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}, step=example_ct)
        logging.info(f"Epoch {epoch}, Validation Acc after {str(example_ct).zfill(5)} examples: {val_acc:.3f}")
        logging.info(f"Epoch {epoch}, Validation F1  after {str(example_ct).zfill(5)} examples: {val_f1:.3f}")
        # fmt: on

    return model


def train_batch(
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
    if config.multitask is True:
        losses = []
        for idx, crit in enumerate(criterion):
            label_col = config.label_cols[idx]
            batch_y = batch_y["label_col"]
            loss = crit(outputs, batch_y)
            losses.append(loss)
        loss = torch.concat(*losses)
        loss = torch.mean(loss)
    else:
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


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    config: SimpleNamespace,
) -> Tuple[float, float, int]:
    """Test model performance over data in test_loader

    Args:
        model (nn.Module): _description_
        test_loader (DataLoader): _description_
        num_classes (int): _description_
        device (torch.device): _description_
        config

    Returns:
        Tuple[float, float, int]:
            test accuracy, test f1 score, and number of samples tested on
    """
    # Device
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

            # Push batch_y to device multitask
            if config.multitask is True:
                for label_col, labels_batch in batch_y.items():
                    batch_y[label_col] = labels_batch.to(device)

            # Push batch_y to device single task (and extract from batch_y dictionary)
            else:
                batch_y = batch_y[config.label_col].to(device)

            # Forward
            input_ids = batch_x.input_ids
            attention_mask = batch_x.attention_mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Predictions: TODO: handle multi-task predictions
            batch_preds = torch.argmax(outputs, dim=-1)

            all_preds.append(batch_preds)
            all_labels.append(batch_y)
            total += batch_y.size(0)

    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)

    # Evaluation
    test_acc = accuracy_torch(all_preds, all_labels, num_classes)
    test_f1 = f1_torch(all_preds, all_labels, num_classes)
    return test_acc, test_f1, total


def save_model(model: nn.Module, loader: DataLoader) -> None:
    for batch_x, _ in loader:
        input_ids = batch_x.input_ids
        attention_mask = batch_x.attention_mask
        break

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, (input_ids, attention_mask), "model.onnx")
    wandb.save("model.onnx")


def make(
    config,
) -> Tuple[
    nn.Module,
    DataLoader,
    DataLoader,
    DataLoader,
    Union[nn.Module, List[nn.Module]],
    torch.optim.Optimizer,
]:
    """
    Make model, train/val/test data loaders, loss criterion, and optimizer

    Args:
        config
    """

    # Make tokenizer
    tokenizer = make_tokenizer()
    logging.info("Loaded tokenizer")
    config.vocab_size = tokenizer.vocab_size

    # Make the data loaders
    train, val, test = make_datasets(config)
    train_loader = make_dataloader(train, tokenizer, batch_size=config.batch_size)
    val_loader = make_dataloader(val, tokenizer, batch_size=config.batch_size)
    test_loader = make_dataloader(test, tokenizer, batch_size=config.batch_size)
    logging.info("Made data loaders")

    # Make the model
    model = make_model(config)
    logging.info("Loaded model")
    model = model.to(config.device)

    # Make the loss and optimizer
    criterion = make_criterion(config)
    optimizer = make_optimizer(config, model)
    logging.info("Created loss criterion and optimizer")

    return model, train_loader, val_loader, test_loader, criterion, optimizer


def model_pipeline(config: Dict) -> nn.Module:
    """Load model, train model, evaluate model, and save trained model

    Args:
        config (Dict): dictionary of configurations

    Returns:
        nn.Module: trained model
    """
    project_name = config["project_name"]

    # tell wandb to get started
    with wandb.init(project=project_name, config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        # fmt: off
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config)
        logging.info("Finished make() function")
        logging.info(model)
        # fmt: on

        # and use them to train the model
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer, config
        )
        logging.info("Finished train() function")

        # fmt: off
        # and test its final performance
        test_acc, test_f1, sample_count = test_model(model, test_loader, config.num_classes, config.device, config)
        logging.info(f"Test Accuracy on the {str(sample_count).zfill(5)} test samples: {test_acc:.3f}")
        logging.info(f"Test F1       on the {str(sample_count).zfill(5)} test samples: {test_f1:.3f}")
        wandb.log({"test_acc": test_acc, "test_f1": test_f1})
        # fmt: on

        # Save model
        save_model(model, test_loader)

    return model


if __name__ == "__main__":

    # fmt: off
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, default=None)
    parser.add_argument("--test_run", default=False)
    parser.add_argument("--test_run_n_samples", default=30)
    args = parser.parse_args()
    config_filepath = args.config_filepath
    test_run = bool(args.test_run)
    test_run_n_samples = int(args.test_run_n_samples)
    # fmt: on

    # Wandb setup
    wandb.login()

    # Load config
    config = load_config(config_filepath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    logging.info("Loaded config:")
    logging.info(pformat(config))

    # Test run setup
    if test_run is True:
        config["test_run"] = True
        config["test_run_n_samples"] = test_run_n_samples

    model_pipeline(config)
