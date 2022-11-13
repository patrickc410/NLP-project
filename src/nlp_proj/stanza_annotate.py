""" 
Runs Stanza client annotation over dataset, then saves results
to a JSON Lines file

Usage:

python src/nlp_proj/stanza_annotate.py \
    --data_filepath ../wiki-auto/wiki-manual/test.tsv \
    --out_filepath ./data/stanza_annotate/test_annotations.jsonl \
    --test_run  True \
    --port 3002

python src/nlp_proj/stanza_annotate.py \
    --data_filepath ../wiki-auto/wiki-manual/train.tsv \
    --out_filepath ./data/stanza_annotate/train_annotations.jsonl \
    --port 3003

python src/nlp_proj/stanza_annotate.py \
    --data_filepath ../wiki-auto/wiki-manual/dev.tsv \
    --out_filepath ./data/stanza_annotate/dev_annotations2.jsonl

"""
import sys
import argparse
from typing import Tuple, Union, List, Dict
import logging
import pathlib

import stanza
from stanza.server import CoreNLPClient
import pandas as pd
from datasets import Dataset

from nlp_proj.data_loader import (
    load_wiki_manual_tsv,
    concat_wiki_manual_sentences,
)

logging.getLogger().setLevel(logging.INFO)
SVO_TYPE = List[Tuple[str, int]]


def get_tokens(sentence_dict: Dict) -> Tuple[List[str], List[int]]:
    """Get list of (token, token index) tuples from an annotated sentence dictionary

    Args:
        sentence_dict (Dict): dictionary from document["sentences"] after client.annotate

    Returns:
        Tuple[List[str], List[int]]: _description_
    """

    if "tokens" not in sentence_dict.keys():
        return [], []

    tokens = [entry["word"] for entry in sentence_dict["tokens"]]
    token_idxs = [(entry["index"]) for entry in sentence_dict["tokens"]]
    return tokens, token_idxs


def get_subj_verb_obj(sentence_dict: Dict) -> Tuple[SVO_TYPE, SVO_TYPE, SVO_TYPE]:
    """Get the subj, verb, obj from a given annotated sentence dictionary

    Args:
        sentence_dict (Dict): dictionary from document["sentences"] after client.annotate

    Returns:
        Tuple[SVO_TYPE, SVO_TYPE, SVO_TYPE]: tuple of subj, verb, obj
            where each is either None or a tuple of the string and its index position in the sentence
    """

    subj = set()
    verb = set()
    obj = set()

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
                if dep["dep"] in ["nsubj", "nsubj:pass"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    verb.add((verb_str, verb_idx))
                    subj_str = dep["dependentGloss"]
                    subj_idx = dep["dependent"]
                    subj.add((subj_str, subj_idx))

                if dep["dep"] in ["aux:pass"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    verb.add((verb_str, verb_idx))
                    verb_str = dep["dependentGloss"]
                    verb_idx = dep["dependent"]
                    verb.add((verb_str, verb_idx))

                elif dep["dep"] == "obj":
                    if obj is None:
                        obj_str = dep["dependentGloss"]
                        obj_idx = dep["dependent"]
                        obj = (obj_str, obj_idx)

        idx += 1

    return sorted(subj), sorted(verb), sorted(obj)


def determine_passive(sentence_dict: Dict) -> Tuple[bool, SVO_TYPE]:
    """Determine if sentence has passive voice and return list of passive voice verbs
    used if applicable

    Args:
        sentence_dict (Dict): dictionary from document["sentences"] after client.annotate

    Returns:
        Tuple[bool, SVO_TYPE]: _description_
    """

    passive_verb = set()

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

                if dep["dep"] in ["aux:pass"]:
                    verb_str = dep["governorGloss"]
                    verb_idx = dep["governor"]
                    passive_verb.add((verb_str, verb_idx))
                    verb_str = dep["dependentGloss"]
                    verb_idx = dep["dependent"]
                    passive_verb.add((verb_str, verb_idx))

        idx += 1

    return len(passive_verb) != 0, sorted(passive_verb)


# def parse_syntax_tree_str(tree: str):
#     """_summary_

#     Example: (ROOT\n  (S\n    (NP (DT The) (NNP Gandalf) (NNPS Awards))\n    (VP (VBP honor)\n      (NP (JJ excellent) (NN writing))\n      (PP (IN in)\n        (PP (IN in)\n          (NP (NN fantasy) (NN literature)))))\n    (. .)))

#     ('(ROOT\n'
#     '  (S\n'
#     '    (NP (DT The) (NNP Gandalf) (NNPS Awards))\n'
#     '    (VP (VBP honor)\n'
#     '      (NP (JJ excellent) (NN writing))\n'
#     '      (PP (IN in)\n'
#     '        (PP (IN in)\n'
#     '          (NP (NN fantasy) (NN literature)))))\n'
#     '    (. .)))')

#     Args:
#         tree (str): _description_
#     """
#     pass


def dataset_annotate_sentence(dataset_row: Dict, client: CoreNLPClient) -> Dict:
    """CoreNLPClient annotation on a sentence. Use num_proc=1 to avoid deadlocks
    Updates dataset_row dictionary

    Args:
        dataset_row (Dict): Dataset row
        client (CoreNLPClient):

    Returns:
        Dict: updated Dataset row
    """
    sent = dataset_row["sent"]
    document = client.annotate(sent, output_format="json")
    sentence_dict = document["sentences"][0]
    dataset_row["sentence_dict"] = sentence_dict
    return dataset_row


# def dataset_get_subj_verb_obj(dataset_row: Dict) -> Dict:
#     """Wrapper function to extract subj, verb, obj from annotated sentence dictionary
#     and update dataset_row dictionary

#     Args:
#         dataset_row (Dict):  Dataset row

#     Returns:
#         Dict: updated Dataset row
#     """
#     sentence_dict = dataset_row["sentence_dict"]
#     svo = get_subj_verb_obj(sentence_dict)
#     dataset_row["subj"] = svo[0][0] if svo[0] is not None else ""
#     dataset_row["subj_idx"] = svo[0][1] if svo[0] is not None else -1
#     dataset_row["verb"] = svo[1][0] if svo[1] is not None else ""
#     dataset_row["verb_idx"] = svo[1][1] if svo[1] is not None else -1
#     dataset_row["obj"] = svo[2][0] if svo[2] is not None else ""
#     dataset_row["obj_idx"] = svo[2][1] if svo[2] is not None else -1
#     return dataset_row


# def dataset_get_tokens(dataset_row: Dict) -> Dict:
#     """Wrapper function to extract tokens from annotated sentence dictionary
#     and update dataset_row dictionary

#     Args:
#         dataset_row (Dict): Dataset row

#     Returns:
#         Dict: updated Dataset row
#     """
#     sentence_dict = dataset_row["sentence_dict"]
#     tokens = get_tokens(sentence_dict)
#     dataset_row["tokens"] = tokens
#     return dataset_row


if __name__ == "__main__":

    # fmt: off
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, default="../wiki-auto/wiki-manual/dev.tsv")
    parser.add_argument("--out_filepath", type=str, default="./data/stanza_annotate/dev_annotations.jsonl")
    parser.add_argument("--test_run", default=False)
    parser.add_argument("--port", type=int, default=3002)
    args = parser.parse_args()
    data_filepath = args.data_filepath
    out_filepath = args.out_filepath
    test_run = bool(args.test_run)
    port = int(args.port)
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

    # Setup Core NLP tools -- only necessary once
    stanza.download("en")
    stanza.install_corenlp()

    # Setup data
    df, columns = load_wiki_manual_tsv(data_filepath)
    df_sent = concat_wiki_manual_sentences(df, drop_duplicates=True)
    if test_run is True:
        df_sent = df_sent.sample(n=6000)
        logging.info("TEST RUN, sampling data to 10 rows")
    dataset = Dataset.from_pandas(df_sent)
    logging.info("Loaded data:")
    logging.info(dataset)

    # Set up NLP client
    client = CoreNLPClient(
        timeout=150000000,
        be_quiet=True,
        annotators=["kbp", "parse"],
        endpoint=f"http://localhost:{port}",
    )
    client.start()

    # fmt: off
    #  CoreNLPClient annotate. Must be done with num_proc=1 to avoid deadlocks; takes a while!
    dataset = dataset.map(lambda row: dataset_annotate_sentence(row, client), num_proc=1)
    client.stop()
    logging.info("Finished CoreNLPClient annotations")
    # fmt: on

    # Save
    dataset.to_json(out_filepath)
