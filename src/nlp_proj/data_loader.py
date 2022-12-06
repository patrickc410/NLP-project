import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Dict
import logging
import numpy as np


def load_wiki_manual_tsv(data_filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load Wiki-Manual TSV file as DataFrame

    Args:
        data_filepath (str): path to .tsv

    Returns:
        Tuple[pd.DataFrame, List[str]]: dataframe, list of column names
    """
    columns = [
        "label",
        "simple-sent-index",
        "complex-sent-index",
        "simple-sent",
        "complex-sent",
        "GLEU-score",
    ]
    try:
        df = pd.read_csv(data_filepath, delimiter="\t", names=columns, header=None)
    except Exception:
        data = []
        error_count = 0
        with open(data_filepath, "r") as f:
            for line in f.readlines():
                split = line.split("\t")
                if len(split) != len(columns):
                    error_count += 1
                else:
                    entry = {col: val for col, val in zip(columns, split)}
                    data.append(entry)
        if error_count > 0:
            logging.warn(
                f"Unable to load {error_count} lines of file '{data_filepath}'"
            )
        df = pd.DataFrame(data)

    return df, columns


def concat_wiki_manual_sentences(
    df: pd.DataFrame, drop_duplicates: bool = True
) -> pd.DataFrame:
    """Make a dataframe with rows of (sent-index, sent)
    where all the ("simple-sent-index", "simple-sent") and ("complex-sent-index", "complex-sent")
    pairs have been concatenated along the same axis

    Args:
        df (pd.DataFrame): dataframe from load_wiki_manual_tsv()
        drop_duplicates (bool): whether to drop duplicates

    Returns:
        pd.DataFrame: _description_
    """
    df_simple = df[["simple-sent-index", "simple-sent"]]
    df_simple = df_simple.rename(
        columns={"simple-sent-index": "sent-index", "simple-sent": "sent"}
    )
    df_complex = df[["complex-sent-index", "complex-sent"]]
    df_complex = df_complex.rename(
        columns={"complex-sent-index": "sent-index", "complex-sent": "sent"}
    )
    df_out = pd.concat((df_simple, df_complex), axis=0, ignore_index=True)
    if drop_duplicates is True:
        df_out = df_out.drop_duplicates(ignore_index=True)
    return df_out


class WikiManualDataset(Dataset):
    """Class for serving every row of Wiki-Manual dataset"""

    def __init__(self, data_filepath: str) -> None:
        self.data_filepath = data_filepath
        self.df, self.columns = load_wiki_manual_tsv(self.data_filepath)

        super().__init__()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> List:
        """Return entire row"""
        return self.df.iloc[index].tolist()


class WikiManualSentenceDataset(WikiManualDataset):
    """Class for serving every individual sentence of Wiki-Manual dataset"""

    def __len__(self) -> int:
        return 2 * super().__len__()

    def __getitem__(self, index: int) -> List:
        """Returns (simple-sent-index, simple-sent) or (complex-sent-index, complex-sent)"""
        df_idx = index // 2
        complex_ind = index % 2 == 0
        row = self.df.iloc[df_idx]
        if complex_ind is True:
            return row[["simple-sent-index", "simple-sent"]].tolist()
        else:
            return row[["complex-sent-index", "complex-sent"]].tolist()


class WikiManualActiveVoiceDataset(Dataset):
    def __init__(
        self,
        data_filepath: str,
        sent_col: str = "sent",
        label_col: str = "active_stanza",
        drop_labels: List[str] = None,
    ) -> None:
        self.data_filepath = data_filepath
        self.df = pd.read_json(data_filepath, lines=True, orient="records")
        self.sent_col = sent_col
        if sent_col not in self.df.columns:
            raise Exception(
                f"Expected column {label_col} in dataset from file {data_filepath}"
            )
        if label_col not in self.df.columns:
            raise Exception(
                f"Expected column {label_col} in dataset from file {data_filepath}"
            )
        self.df = self.df[[sent_col, label_col]]
        if drop_labels is not None:
            num_drops = self.df[self.df[label_col].isin(drop_labels)].shape[0]
            logging.info(
                f"Dropping {num_drops} rows which correspond to drop_labels: {drop_labels}"
            )
            self.df = self.df[~self.df[label_col].isin(drop_labels)]
        self.labels, label_idxs, self.label_counts = np.unique(
            self.df[label_col], return_inverse=True, return_counts=True
        )
        self.df["label_idx"] = label_idxs

        super().__init__()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> List:
        """Return sentence string and active voice label (either "active", "passive", or "both")"""
        row = self.df.iloc[index]
        return row[self.sent_col], row["label_idx"]
