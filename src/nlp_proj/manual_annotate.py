""" 
Command Line tool for manually annotating sentences of the wiki-manual 
dataset after augmentation from stanza client annotation

Usage:

python src/nlp_proj/manual_annotate.py \
    --data_filepath ./data/stanza_annotate/dev_annotations.jsonl \
    --out_dir ./data/manual_annotate/patrick_100_200 \
    --which_annotate subj \
    --annotator patrick \
    --start_idx 100 \
    --end_idx 115

python src/nlp_proj/manual_annotate.py \
    --data_filepath ./data/stanza_annotate/dev_annotations.jsonl \
    --out_dir ./data/manual_annotate/patrick_0_100 \
    --which_annotate active \
    --annotator patrick \
    --start_idx 0 \
    --end_idx 100

"""
import argparse
import pathlib
import sys
import os
import logging
from typing import List, Dict

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
import pandas as pd

from nlp_proj.stanza_annotate import get_tokens, get_subj_verb_obj, SVO_TYPE

logging.getLogger().setLevel(logging.INFO)


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def write_annotation_row(out_filepath: str, annotation_row: Dict) -> None:
    """Write annotation row to out file in append mode

    Args:
        out_filepath (str): _description_
        annotation_row (Dict): _description_
    """
    out_df = pd.DataFrame([annotation_row])
    with open(out_filepath, "a") as f:
        out_df.to_json(f, orient="records", lines=True)


def handle_svo_annotation(
    sent: str,
    sent_index: str,
    tokens: List[str],
    token_idxs: List[int],
    svo: SVO_TYPE,
    which: str,
) -> Dict:
    """Handle prompting user for annotation on subj, verb, obj multi-label annotation

    Args:
        sent (str): the sentence
        sent_index (str): sentence index in dataset
        tokens (List[str]): list of words in sentence
        token_idxs (List[int]): list of word indexes in sentence
        svo (SVO_TYPE): output of get_subj_verb_obj(sentence_dict)
        which (str): which of subj, verb, obj to annotate

    Returns:
        Dict: _description_
    """
    which = which.lower()
    which_long = None
    if which == "subj":
        svo = svo[0]
        which_long = "subject"
    elif which == "verb":
        svo = svo[1]
        which_long = "verb"
    elif which == "obj":
        svo = svo[2]
        which_long = "object"
    else:
        raise Exception(f"Invalid argument which={which}")

    print(f"\nSENTENCE: {sent}")

    svo_ind_list = [False for _ in range(len(tokens))]
    if svo is not None:
        svo_ind_list = [(token_idxs[i] == svo[1]) for i in range(len(tokens))]

    svo_choices = [
        Choice((token, idx), name=token, enabled=svo_ind)
        for token, idx, svo_ind in zip(tokens, token_idxs, svo_ind_list)
    ]
    svo_choices.insert(
        0, Choice(("<DON'T KNOW>", -1), name="<DON'T KNOW>", enabled=False)
    )

    svo_selection = inquirer.checkbox(
        message=f"Select all {which_long} tokens(s):",
        choices=svo_choices,
        cycle=True,
    ).execute()

    annotation_row = {"sent-index": sent_index, f"{which}_manual": svo_selection}
    return annotation_row


def handle_active_voice_annotation(sent: str, sent_index: str) -> Dict:
    """Handle prompting user for annotation of active vs. passive voice

    Args:
        sent (str): the sentence
        sent_index (str): sentence index in dataset

    Returns:
        Dict: _description_
    """
    print(f"\nSENTENCE: {sent}")

    active_voice_choices = [
        Choice(-1, name="<DON'T KNOW>", enabled=False),
        Choice(1, name="active voice", enabled=False),
        Choice(0, name="passive voice", enabled=False),
    ]

    active_voice_selection = inquirer.select(
        message=f"Select whether the sentence uses active or passive voice: ",
        choices=active_voice_choices,
        cycle=True,
    ).execute()

    annotation_row = {
        "sent-index": sent_index,
        "active_voice_manual": active_voice_selection,
    }
    return annotation_row


if __name__ == "__main__":

    # fmt: off
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, default="../../data/stanza_annotate/dev_annotations.jsonl")
    parser.add_argument("--out_dir", type=str, default="../../data/manual_annotate/")
    parser.add_argument("--which_annotate", type=str, default=None)
    parser.add_argument("--annotator", type=str)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    # fmt: on

    # Parse command line arguments
    args = parser.parse_args()
    data_filepath = args.data_filepath
    out_dir = args.out_dir
    which_annotate = args.which_annotate.lower()
    annotator = args.annotator.lower()
    arg_start_idx = args.start_idx
    arg_end_idx = args.end_idx

    # Validate annotator arg
    if annotator is None:
        raise Exception("Must provide an annotator name")
    logging.info(f"Annotator name: {annotator}")

    # Check which_annotate arg
    supported_annotations = ["subj", "verb", "obj", "active"]
    if which_annotate not in supported_annotations:
        raise Exception(f"Support for annotating {which_annotate} not implemented yet")
    logging.info(f"Which annotation: {which_annotate}")

    # Validate data_filepath
    if not pathlib.Path(data_filepath).is_file():
        p = str(pathlib.Path(data_filepath).resolve())
        raise Exception(f"Provided data_filepath '{p}' is not a file")
    logging.info(f"Using data_filepath: {data_filepath}")

    # Validate out_dir
    if not pathlib.Path(out_dir).is_dir():
        print(f"Provided out_dir '{out_dir}' does not exist and will be created. ")
        res = input("Proceed? [y/n] ").lower()
        if res != "y":
            sys.exit()
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build out_filepath
    dpath = pathlib.Path(data_filepath)
    stem = dpath.stem
    out_filepath = pathlib.Path(out_dir, f"{stem}_{which_annotate}.jsonl")
    logging.info(f"Using out_filepath: '{str(out_filepath.resolve())}'")

    # Check if resuming annotation
    resume_from_last_annotation = False
    if pathlib.Path(out_filepath).is_file():
        print(f"out_filepath file '{out_filepath}' already exists. ")
        res = input("Continue from last annotation? [y/n] ")
        if res == "y":
            resume_from_last_annotation = True
    logging.info(f"Continue from last annotation = {resume_from_last_annotation}")

    # Load data
    df = pd.read_json(data_filepath, lines=True, orient="records")
    logging.info(f"Loaded dataset of shape {df.shape}, with columns {df.columns}")

    # Determine start index
    start_idx = 0
    if arg_start_idx is not None:
        start_idx = int(arg_start_idx)
    if resume_from_last_annotation is True:
        df_anno = pd.read_json(out_filepath, lines=True, orient="records")
        last_anno_row = df_anno.tail(1)
        last_anno_sent_index = last_anno_row["sent-index"].item()
        df_last_anno_row = df[df["sent-index"] == last_anno_sent_index].tail(1)
        df_last_anno_idx = df_last_anno_row.index.item()
        start_idx = df_last_anno_idx + 1
    logging.info(f"Using starting index: {start_idx}")

    # Determine end index
    end_idx = len(df)
    if arg_end_idx is not None:
        end_idx = min(int(arg_end_idx), df.index.max())
    if end_idx < start_idx:
        raise Exception(
            f"ending index ({end_idx}) must be greater than or equal to starting index ({start_idx}) "
        )
    logging.info(f"Using ending index: {end_idx}")

    # Get subset to annotate
    # fmt: off
    df_subset = df[start_idx:end_idx]
    logging.info(f"Using dataset subset of shape {df_subset.shape} with start index {df_subset.index.min()} and end index {df_subset.index.max()}")
    # fmt: on

    for idx, row in df_subset.iterrows():

        sent = row["sent"]
        sent_index = row["sent-index"]
        sentence_dict = row["sentence_dict"]

        tokens, token_idxs = get_tokens(sentence_dict)
        if len(tokens) == 0:
            logging.info(f"Issue getting tokens for row idx={idx}")
        svo = get_subj_verb_obj(sentence_dict)

        if which_annotate == "subj":
            anno_row = handle_svo_annotation(
                sent, sent_index, tokens, token_idxs, svo, which="subj"
            )
        elif which_annotate == "verb":
            anno_row = handle_svo_annotation(
                sent, sent_index, tokens, token_idxs, svo, which="verb"
            )
        elif which_annotate == "obj":
            anno_row = handle_svo_annotation(
                sent, sent_index, tokens, token_idxs, svo, which="obj"
            )
        elif which_annotate == "active":
            anno_row = handle_active_voice_annotation(sent, sent_index)

        anno_row["annotator"] = annotator

        write_annotation_row(out_filepath, anno_row)

    logging.info("FINISHED annotations")
