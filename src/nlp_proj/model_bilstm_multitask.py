import torch
import torch.nn as nn
from typing import List
from torch import Tensor


class BiLSTMMultitask(nn.Module):
    def __init__(
        self,
        num_classes_list: List[int],
        vocab_size: int,
        dim_emb: int = 256,
        dim_hid: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.dim_emb = dim_emb
        self.dim_hid = dim_hid
        self.num_classes_list = num_classes_list
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
        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(self.num_classes_list):
            head = self.make_head(self.dim_hid, num_classes)
            self.heads.add_module(f"head{i}", head)

    def make_head(self, dim_hid: int, num_classes: int) -> nn.Sequential:
        """Make classification/regression head"""
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hid, dim_hid),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim_hid, num_classes),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> List[Tensor]:
        # fmt: off
        x = self.embed(input_ids)           # (N, SLEN, DIM_EMB)
        out, (h_n, c_n) = self.lstm(x)      # out: (N, SLEN, 2*DIM_HID), h_n: (2*num_layers, N, DIM_HID)
        h_n = h_n.permute((1, 0, 2))        # (N, 2*num_layers, DIM_HID)
        pooled = torch.mean(h_n, 1)         # (N, DIM_HID)
        outputs = []
        for head in self.heads:
            outputs.append(head(pooled))    # (N, num_classes)
        return outputs
        # fmt: on
