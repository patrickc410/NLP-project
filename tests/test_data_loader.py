#!/usr/bin/python3

from pathlib import Path

from nlp_proj.data_loader import WikiManualDataset, WikiManualSentenceDataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_WikiManualDataset():

    dataset = WikiManualDataset("../wiki-auto/wiki-manual/dev.tsv")
    assert dataset is not None
