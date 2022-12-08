import transformers
import torch
import torch.nn as nn
from torch import Tensor


class BERTClassifier(nn.Module):
    def __init__(
        self, num_classes: int, dim_hid: int = 20, freeze_pretrained: bool = False
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim_hid = dim_hid
        self.freeze_pretrained = freeze_pretrained
        self.pretrained_layers = transformers.AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        if self.freeze_pretrained:
            for param in self.pretrained_layers.parameters():
                param.requires_grad = False

        self.head = nn.Sequential()
        self.head.add_module("fc_1", nn.Linear(768, dim_hid))
        self.head.add_module("relu_1", nn.ReLU())
        self.head.add_module("dropout", nn.Dropout(0.1))
        self.head.add_module("fc_2", nn.Linear(dim_hid, 3))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Use last hidden state of first token as the "pooled" output of the pre-trained model,
        as is done in Huggingface implementation https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
        """
        pretrained_out = self.pretrained_layers(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (bs, seq_len, dim)
        pooled_output = pretrained_out[:, 0]  # (bs, dim)
        x = self.head(pooled_output)  # (bs, 1)
        return x


class BERTRegressor(BERTClassifier):
    def __init__(self, dim_hid: int = 20, freeze_pretrained: bool = False) -> None:
        super().__init__(1, dim_hid, freeze_pretrained)
