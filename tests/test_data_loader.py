#!/usr/bin/python3

from pathlib import Path

from nlp_proj.data_loader import (
    function
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_function():
    assert function() is None