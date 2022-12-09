""" 


Usage

python src/nlp_proj/stanza_annotate_active_voice.py \
    --data_filepath ./data/stanza_annotate/dev_annotations.jsonl \
    --out_filepath ./data/stanza_annotate/dev_annotations_active.jsonl \
    --test_run True \
    --test_run_n_samples 30

python src/nlp_proj/stanza_annotate_active_voice.py \
    --data_filepath ./data/stanza_annotate/dev_annotations.jsonl \
    --out_filepath ./data/stanza_annotate/dev_annotations_active.jsonl

"""
import argparse
import pathlib
import logging
import sys
from typing import Dict, Tuple

from tqdm import tqdm
import pandas as pd

from nlp_proj.stanza_annotate import determine_passive, SVO_TYPE

logging.getLogger().setLevel(logging.INFO)


def determine_active_passive_label(
    sentence_dict: Dict,
) -> Tuple[str, SVO_TYPE, SVO_TYPE]:
    """Determine if sentence has passive voice and return list of passive voice verbs
    used if applicable

    Args:
        sentence_dict (Dict): dictionary from document["sentences"] after client.annotate

    Returns:
        Tuple[str, SVO_TYPE, SVO_TYPE]
        str: a string either "active", "passive", or "both" where the labels are as follows:
            "active"    all clauses are active voice
            "passive":  all clauses are passive voice
            "both":     there exists at least one clause with active voice and one clause with passive voice
        SVO_TYPE: a list of active verbs as tuples of the verb's string and its index position in the sentence
        SVO_TYPE: a list of passive verbs as tuples of the verb's string and its index position in the sentence
    """

    passive_verb = set()
    active_verb = set()

    dep_keys = [
        "basicDependencies",
        "enhancedDependencies",
        "enhancedPlusPlusDependencies",
    ]
    idx = 0
    while idx < len(dep_keys):
        dependency_key = dep_keys[idx]
        if dependency_key in sentence_dict.keys():
            deps = sentence_dict[dependency_key]

            for dep in deps:
                if dep["dep"] in ["nsubj:pass"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    passive_verb.add((verb_str, verb_idx))

                if dep["dep"] in ["nsubj"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    active_verb.add((verb_str, verb_idx))

                if dep["dep"] in ["aux:pass"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    passive_verb.add((verb_str, verb_idx))
                    verb_str = dep["dependentGloss"]
                    verb_idx = dep["dependent"]
                    passive_verb.add((verb_str, verb_idx))

        idx += 1

    label = "<DON'T KNOW>"
    if len(active_verb) > 0 and len(passive_verb) > 0:
        label = "both"
    elif len(active_verb) > 0:
        label = "active"
    elif len(passive_verb) > 0:
        label = "passive"
    else:
        label = "<DON'T KNOW>"

    return label, sorted(active_verb), sorted(passive_verb)


if __name__ == "__main__":

    # fmt: off
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, default="../wiki-auto/wiki-manual/dev.tsv")
    parser.add_argument("--out_filepath", type=str, default="./data/stanza_annotate/dev_annotations.jsonl")
    parser.add_argument("--test_run", default=False)
    parser.add_argument("--test_run_n_samples", default=10)
    args = parser.parse_args()
    data_filepath = args.data_filepath
    out_filepath = args.out_filepath
    test_run = bool(args.test_run)
    test_run_n_samples = int(args.test_run_n_samples)
    # fmt: on

    # Validate command line args
    if not pathlib.Path(data_filepath).is_file():
        p = str(pathlib.Path(data_filepath).resolve())
        raise Exception(f"Provided data_filepath '{p}' is not a file")
    if not pathlib.Path(out_filepath).parent.is_dir():
        out_dir = pathlib.Path(out_filepath).parent.resolve()
        o = str(out_dir)
        print(f"Provided out_filepath dir '{o}' does not exist and will be created. ")
        res = input("Proceed? [y/n] ").lower()
        if res != "y":
            sys.exit()
        out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using data_filepath: '{data_filepath}'")
    logging.info(f"Using out_filepath: '{out_filepath}'")

    # Setup data
    # Load data
    df = pd.read_json(data_filepath, lines=True, orient="records")
    logging.info(f"Loaded dataset of shape {df.shape}, with columns {df.columns}")
    if test_run is True:
        df = df.sample(n=test_run_n_samples)
        logging.info(f"TEST RUN, sampling data to {test_run_n_samples} rows")
    logging.info(f"Loaded data of shape {df.shape}:")
    logging.info(df.head())

    def wrapper_determine_active_passive_label(row: pd.Series) -> str:
        sentence_dict = row["sentence_dict"]
        return determine_active_passive_label(sentence_dict)

    # fmt: off
    tqdm.pandas()
    new_columns: pd.DataFrame = df.progress_apply(wrapper_determine_active_passive_label, axis="columns", result_type="expand")
    new_columns = new_columns.rename(columns={0: "active_stanza", 1: "active_verbs_stanza", 2: "passive_verbs_stanza"})
    df = pd.concat([df, new_columns], axis='columns')
    logging.info("Finished determining active/passive labels")
    # fmt: on

    # Save
    df.to_json(out_filepath, orient="records", lines=True)
