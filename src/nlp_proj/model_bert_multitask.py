import transformers
import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class BERTMultitask(nn.Module):
    def __init__(
        self,
        num_classes_list: List[int],
        dim_hid: int = 20,
        freeze_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes_list = num_classes_list
        self.dim_hid = dim_hid
        self.freeze_pretrained = freeze_pretrained
        self.pretrained_layers = transformers.AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        if self.freeze_pretrained:
            for param in self.pretrained_layers.parameters():
                param.requires_grad = False

        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(self.num_classes_list):
            head = self.make_head(self.dim_hid, num_classes)
            self.heads.add_module(f"head{i}", head)

    def make_head(self, dim_hid: int, num_classes: int) -> nn.Sequential:
        head = nn.Sequential()
        head.add_module("fc_1", nn.Linear(768, dim_hid))
        head.add_module("relu_1", nn.ReLU())
        head.add_module("dropout", nn.Dropout(0.1))
        head.add_module("fc_2", nn.Linear(dim_hid, num_classes))
        return head

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> List[Tensor]:
        """Use last hidden state of first token as the "pooled" output of the pre-trained model,
        as is done in Huggingface implementation https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
        """
        pretrained_out = self.pretrained_layers(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (bs, seq_len, dim)
        pooled_output = pretrained_out[:, 0]  # (bs, dim)
        outputs = []
        for head in self.heads:
            outputs.append(head(pooled_output))  # (N, num_classes)
        return outputs
