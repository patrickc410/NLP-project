import torch
import torch.nn as nn
from torch import Tensor


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        dim_emb: int = 256,
        dim_hid: int = 20,
        num_layers: int = 2,
    ):
        """_summary_

        Args:
            num_classes (int): number of output classes
            vocab_size (int): size of vocab for embedding layer
            dim_emb (int, optional): size of embedding layer. Defaults to 256.
            dim_hid (int, optional): size of hidden state dimension in LSTM cells. Defaults to 20.
            num_layers (int, optional): number of LSTM layers. Defaults to 2.
        """
        super().__init__()
        self.dim_emb = dim_emb
        self.dim_hid = dim_hid
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(self.vocab_size, self.dim_emb)
        self.lstm = nn.LSTM(
            input_size=self.dim_emb,
            hidden_size=self.dim_hid,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.head = self.make_head(self.dim_hid, self.num_classes)

    def make_head(self, dim_hid: int, num_classes: int) -> nn.Sequential:
        """Make classification/regression head"""
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hid, dim_hid),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim_hid, num_classes),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None):
        # fmt: off
        x = self.embed(input_ids)           # (N, SLEN, DIM_EMB)
        out, (h_n, c_n) = self.lstm(x)      # out: (N, SLEN, 2*DIM_HID), h_n: (2*num_layers, N, DIM_HID)
        h_n = h_n.permute((1, 0, 2))        # (N, 2*num_layers, DIM_HID)
        pooled = torch.mean(h_n, 1)         # (N, DIM_HID)
        output = self.head(pooled)          # (N, num_classes)
        return output
        # fmt: on


class BiLSTMRegressor(BiLSTMClassifier):
    def __init__(
        self,
        vocab_size: int,
        dim_emb: int = 256,
        dim_hid: int = 20,
        num_layers: int = 2,
    ):
        """Same as BiLSTMClassifier, but always has a single output node"""
        super().__init__(1, vocab_size, dim_emb, dim_hid, num_layers)
