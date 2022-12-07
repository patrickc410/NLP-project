""" 
Define Dataset, function to get DataLoader
"""
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Union, Dict, Tuple
import logging
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
import transformers
from types import SimpleNamespace


class WikiManualAllLabelsDataset(Dataset):
    def __init__(
        self,
        data_filepath: str,
        sent_col: str = "sent",
        label_cols: Union[List[str], None] = None,
        drop_labels_list: Union[List[List], None] = None,
    ) -> None:
        """_summary_

        Args:
            data_filepath (str): _description_
            sent_col (str, optional): _description_. Defaults to "sent".
            label_cols (Union[List[str], None], optional): _description_. Defaults to None.
            drop_labels_list (Union[List[List], None], optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
            Exception: _description_
        """
        super().__init__()
        self.data_filepath = data_filepath
        self.df = pd.read_csv(data_filepath)
        self.sent_col = sent_col
        self.label_cols = label_cols
        self.drop_labels_list = drop_labels_list

        # Verify sentence string column exists
        if sent_col not in self.df.columns:
            raise Exception(
                f"Expected column {sent_col} in dataset from file {data_filepath}"
            )

        # Get default label columns
        if label_cols is None:
            label_cols = ["svo_dist", "apv", "scv", "hv", "svo_dist_norm"]
            self.label_cols = label_cols

        # Verify label columns exist
        for label_col in label_cols:
            if label_col not in self.df.columns:
                raise Exception(
                    f"Expected column {label_col} in dataset from file {data_filepath}"
                )

        # Keep only the sentence column and label columns
        self.df = self.df[[sent_col, *label_cols]]

        # Drop labels if provided
        if drop_labels_list is not None:
            if len(drop_labels_list) != len(label_cols):
                raise Exception(
                    f"Expected drop_labels ({drop_labels_list}) to be the same length as label_cols ({label_cols})"
                )
            for drop_labels, label_col in zip(drop_labels_list, label_cols):
                num_drops = self.df[self.df[label_col].isin(drop_labels)].shape[0]
                logging.info(
                    f"Dropping {num_drops} rows which correspond to drop_labels: {drop_labels} for column {label_col}"
                )
                self.df = self.df[~self.df[label_col].isin(drop_labels)]

        # Transform labels into label indexes
        self.labels_list = []
        self.label_idxs_list = []
        self.label_counts_list = []
        for label_col in label_cols:
            labels, label_idxs, label_counts = np.unique(
                self.df[label_col], return_inverse=True, return_counts=True
            )
            self.labels_list.append(labels)
            self.label_counts_list.append(label_counts)
            self.df[f"{label_col}_idx"] = label_idxs

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[str, Dict[str, int]]:
        """Return tuple with (sentence string, dictionary) where the dictionary contains all labels"""
        row = self.df.iloc[index]
        return row[self.sent_col], {
            label_col: row[f"{label_col}_idx"] for label_col in self.label_cols
        }


def make_tokenizer() -> DistilBertTokenizer:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer


def collate_multi_label(
    batch: List[Tuple[str, Dict[str, int]]],
    tokenizer: DistilBertTokenizer,
    label_cols: List[str],
) -> Tuple[transformers.tokenization_utils_base.BatchEncoding, Dict[str, Tensor]]:
    """Collate function for multi-task

    Args:
        batch (List[Tuple[str, Dict[str, int]]]): list of entries returned from WikiManualAllLabelsDataset
        tokenizer (DistilBertTokenizer)

    Returns:
        Tuple[transformers.tokenization_utils_base.BatchEncoding, Dict[str, Tensor]]: _description_
    """
    # Get sentences, convert to tensor with tokenizer
    batch_text = [text for text, _ in batch]
    batch_x = tokenizer(batch_text, return_tensors="pt", padding="longest")

    # Get label dictionaries for each sample of the batch
    batch_label_dicts = [label_dict for _, label_dict in batch]

    # Loop over label columns, getting list of label values
    batch_labels = {}
    for c in label_cols:
        batch_labels[c] = [label_dict[c] for label_dict in batch_label_dicts]

    # Using dict comprehension, cast lists of labels to Tensor
    batch_y = {c: torch.tensor(labels_list) for c, labels_list in batch_labels.items()}

    return batch_x, batch_y


def collate_single_label(
    batch: List[Tuple[str, int]], tokenizer: DistilBertTokenizer, label_col: str
) -> Tuple[transformers.tokenization_utils_base.BatchEncoding, Tensor]:
    """Collate for single label task"""
    batch_text = [text for text, _ in batch]
    batch_label_idxs = [label_dict[label_col] for _, label_dict in batch]
    batch_x = tokenizer(batch_text, return_tensors="pt", padding="longest")
    batch_y = torch.tensor(batch_label_idxs)
    return batch_x, batch_y


def make_dataloader(
    dataset: WikiManualAllLabelsDataset,
    tokenizer: DistilBertTokenizer,
    config: SimpleNamespace,
):
    """Make dataloader, using different collate function on single vs. multitask"""
    if config.multitask is True:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            collate_fn=lambda batch: collate_multi_label(
                batch, tokenizer, config.label_cols
            ),
            shuffle=config.shuffle,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            collate_fn=lambda batch: collate_single_label(
                batch, tokenizer, config.label_col
            ),
            shuffle=config.shuffle,
        )


def make_datasets(config: SimpleNamespace) -> Tuple[Dataset, Dataset, Dataset]:
    """Make train, validation, and test datasets"""

    # Set up basic dataset parameters
    dataset_params = dict()
    if hasattr(config, "sent_col"):
        dataset_params["sent_col"] = config.sent_col
    if hasattr(config, "label_cols"):
        dataset_params["label_cols"] = config.label_cols
    if hasattr(config, "label_col"):
        dataset_params["label_cols"] = [config.label_col]
    if hasattr(config, "drop_labels_list"):
        dataset_params["drop_labels_list"] = config.drop_labels_list

    # If train/val/test dataset paths are all separate
    if all(
        [
            hasattr(config, "train_data_filepath"),
            hasattr(config, "val_data_filepath"),
            hasattr(config, "test_data_filepath"),
        ]
    ):

        train = WikiManualAllLabelsDataset(
            data_filepath=config.train_data_filepath, **dataset_params
        )
        val = WikiManualAllLabelsDataset(
            data_filepath=config.val_data_filepath, **dataset_params
        )
        test = WikiManualAllLabelsDataset(
            data_filepath=config.test_data_filepath, **dataset_params
        )
        logging.info(
            f"Loaded train (length {len(val)}), validation (length {len(val)}), and test (length {len(test)}) data"
        )

    # Otherwise, split the dataset into test/train/val
    else:
        dataset_splits = [0.7, 0.1, 0.2]
        if hasattr(config, "dataset_splits"):
            dataset_splits = config.dataset_splits

        dataset = WikiManualAllLabelsDataset(
            data_filepath=config.data_filepath, **dataset_params
        )
        logging.info(f"Loaded all data of length {len(dataset)}")

        train, val, test = random_split(
            dataset,
            dataset_splits,
            generator=torch.Generator().manual_seed(config.random_seed),
        )
        logging.info(
            f"Made train (length {len(val)}), validation (length {len(val)}), and test (length {len(test)}) data split"
        )

    # Truncate for test runs
    if config.test_run is True:
        train.df = train.df.sample(n=config.test_run_n_samples, random_state=42)
        val.df = val.df.sample(n=config.test_run_n_samples, random_state=42)
        test.df = test.df.sample(n=config.test_run_n_samples, random_state=42)
        logging.info(f"TEST RUN truncating data to length {len(train)}")

    return train, val, test
