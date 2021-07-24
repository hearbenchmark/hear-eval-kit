#!/usr/bin/env python3
"""
Attached labels to all task embeddings.

TODO: Consider combining this with task_embeddings.py. A lot of the
code is shared, and there should be util functions.
"""

import glob
import json
import os.path
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

# This could instead be something from the participants
# TODO: Command line or config?
EMBEDDING_PIP = "hearbaseline"
EMBEDDING_MODEL_PATH = ""  # Baseline doesn't load model
# TODO: Actually load baseline model

EMBED = import_module(EMBEDDING_PIP)


class EmbeddingDataset(IterableDataset):
    """
    Read in all audio file embeddings, and find the labels in the
    associated metadata.

    We assume that this dataset might be too big to fit into memory,
    e.g. for large tasks.

    Since for timestamp embeddings, we might have multiple embeddings
    in a file, we don't really know the size of this dataset in
    advance. Thus we use an iterable dataset. Nonetheless, it might
    be useful to write this dataset to a single file that we stream.
    We might also consider having separate datasets for scene and
    timestamp tasks.

    NOTES:
        * This proceeds embedding first, then label. So models with
        finer-grained hop size will have more examples.
        * This won't work for JND or ranking tasks yet.

    This is one of a handful of design decisions we've made in HEAR.
    """

    def __init__(self, csv_file):
        self.rows = pd.read_csv(csv_file)
        # Since event detection metadata will have duplicates, we de-dup
        # TODO: This suggests we might want to, instead of using CSV files,
        # have a JSONL for metadata, with one file per line.
        # Then there can be timestamp: list[labels].
        # TODO: Make this a generic util function somewhere?
        # TODO: Is this the right column to dedup on ???
        self.rows = self.rows.sort_values(by="slug").drop_duplicates(
            subset="slug", ignore_index=True
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        return r["slug"]


def task_embeddings_with_labels():
    # TODO: Would be good to include the version here
    # https://github.com/neuralaudio/hear2021-eval-kit/issues/37
    embeddir = os.path.join("embeddings", EMBED.__name__)  # type: ignore

    # TODO: tqdm with task name
    for task in glob.glob("tasks/*"):
        print(f"Task {task}")
        task_config = json.loads(open(os.path.join(task, "task_metadata.json")).read())
        task_type = task_config["task_type"]
        for split in ["train", "valid", "test"]:
            print(f"Getting embeddings for {split} split:")

            metadata_csv = os.path.join(task, f"{split}.csv")
            if not os.path.exists(metadata_csv):
                continue

            metadata = pd.read_csv(metadata_csv)

            outdir = os.path.join(embeddir, task, split)

            for embedding_file in tqdm(list(glob.glob(os.path.join(outdir, "*.npy")))):
                # TODO: This is a pretty gross way of recovering the slug, and I wish there were something cleaner.
                embedding_slug = os.path.split(embedding_file)[1].replace(".npy", "")
                if task_type == "scene_labeling":
                    embedding = np.load(embedding_file, allow_pickle=False)

                    rows = metadata[metadata["slug"] == embedding_slug]
                    assert len(rows) == 1
                    label = rows.iloc[0]["label"]
                    # TODO: Insert somewhere
                    # import IPython;
                    # ipshell = IPython.embed;
                    # ipshell(banner1='ipshell')
                elif task_type == "event_labeling":
                    embeddings, timestamps = np.load(embedding_file, allow_pickle=True)

                    import IPython

                    ipshell = IPython.embed
                    ipshell(banner1="ipshell")
                else:
                    raise ValueError


if __name__ == "__main__":
    task_embeddings_with_labels()
