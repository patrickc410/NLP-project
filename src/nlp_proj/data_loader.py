import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List


class WikiManualDataset(Dataset):
    """Class for serving every row of Wiki-Manual dataset"""

    def __init__(self, data_filepath: str) -> None:
        self.data_filepath = data_filepath

        self.columns = [
            "label",
            "simple-sent-index",
            "complex-sent-index",
            "simple-sent",
            "complex-sent",
            "GLEU-score",
        ]
        self.df = pd.read_csv(
            self.data_filepath, delimiter="\t", names=self.columns, header=None
        )

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

