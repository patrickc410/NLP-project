import wandb
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Tuple, Dict, List, Union
from types import SimpleNamespace
from pprint import pformat
import pathlib
import logging

# from dotenv import load_dotenv

from nlp_proj.model_optim_utils import make_model, make_optimizer, make_criterion
from nlp_proj.config_utils import load_config
from nlp_proj.dataset_utils import make_dataloader, make_tokenizer, make_datasets
from nlp_proj.train_utils_singletask import train_model_singletask, test_model_singletask
from nlp_proj.train_utils_multitask import train_model_multitask, test_model_multitask

# load_dotenv()
logging.getLogger().setLevel(logging.INFO)



def save_model(model: nn.Module, loader: DataLoader, config: SimpleNamespace) -> None:
    """ Save model"""
    for batch_x, _ in loader:
        input_ids = batch_x.input_ids
        attention_mask = batch_x.attention_mask
        break

    # Save the model in the exchangeable ONNX format
    out_path: pathlib.Path = pathlib.Path(config.results_dir, f"{config.project_name}.pt")
    torch.save(model, out_path)
    torch.onnx.export(model, (input_ids, attention_mask), out_path.with_suffix(".onnx"))
    wandb.save(out_path)
    wandb.save(out_path.with_suffix(".onnx"))


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
    train_loader = make_dataloader(train, tokenizer, config)
    val_loader = make_dataloader(val, tokenizer, config)
    test_loader = make_dataloader(test, tokenizer, config)
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

        # Make model, data, criterion, and optimizer
        # fmt: off
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config)
        logging.info("Finished make() function")
        logging.info(model)
        # fmt: on

        # Train model
        # fmt: off
        if config.multitask:
            model = train_model_multitask(model, train_loader, val_loader, criterion, optimizer, config)
            logging.info("Finished train() function")

            # Test
            test_evaluation, sample_count = test_model_multitask(model, test_loader, config)
            log_dict = {}
            for c, metric_dict in test_evaluation.items():
                for metric_name, value in metric_dict.items():
                    log_dict[f"test_{c}_{metric_name}"] = value
                    logging.info(f"         Test {c.ljust(13)} {metric_name.ljust(6)} on the {str(sample_count).zfill(5)} test samples: {value:.3f}")
            wandb.log(log_dict)
        
        # Single-task
        else:
            # Train
            model = train_model_singletask(model, train_loader, val_loader, criterion, optimizer, config)
            logging.info("Finished train() function")

            # Test
            # fmt: off
            test_evaluation, sample_count = test_model_singletask(model, test_loader, config)
            log_dict = {}
            for metric_name, value in test_evaluation.items():
                log_dict[f"test_{metric_name}"] = value
                logging.info(f"         Test {metric_name.ljust(6)} on the {str(sample_count).zfill(5)} test samples: {value:.3f}")
            wandb.log(log_dict)
            # fmt: on

        # Save model
        save_model(model, test_loader, config)

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
