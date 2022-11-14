import transformers
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from nlp_proj.data_loader import WikiManualActiveVoiceDataset
from typing import List, Tuple


def make_tokenizer() -> DistilBertTokenizer:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer


def collate(
    batch: List[Tuple[str, int]], tokenizer: DistilBertTokenizer
) -> Tuple[transformers.tokenization_utils_base.BatchEncoding, Tensor]:
    batch_text = [text for text, _ in batch]
    batch_label_idxs = [label_idx for _, label_idx in batch]
    batch_x = tokenizer(batch_text, return_tensors="pt", padding="longest")
    batch_y = torch.tensor(batch_label_idxs)
    return batch_x, batch_y


def make_dataloader(
    dataset: WikiManualActiveVoiceDataset,
    tokenizer: DistilBertTokenizer,
    batch_size: int = 32,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )


class ActiveVoiceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained_layers = transformers.AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        for param in self.pretrained_layers.parameters():
            param.requires_grad = False

        self.classification_layers = nn.Sequential()
        self.classification_layers.add_module("fc_1", nn.Linear(768, 20))
        self.classification_layers.add_module("relu_1", nn.LeakyReLU())
        self.classification_layers.add_module("dropout", nn.Dropout(0.1))
        self.classification_layers.add_module("fc_2", nn.Linear(20, 3))

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Use last hidden state of first token as the "pooled" output of the pre-trained model,
        as is done in Huggingface implementation https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
        """
        pretrained_out = self.pretrained_layers(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (bs, seq_len, dim)
        pooled_output = pretrained_out[:, 0]  # (bs, dim)
        x = self.classification_layers(pooled_output)  # (bs, 1)
        return x

    def predict_pipeline(
        self, sentence: str, tokenizer: transformers.DistilBertTokenizer
    ):
        encoded_input = tokenizer(sentence, return_tensors="pt")
        output = self.forward(**encoded_input)
        pred = torch.round(output)
        return pred
