#!/usr/bin/env python3
"""
Map embeddings to predictions for every downstream task and store
test predictions to disk.

NOTE: Right now this is just a random projection. Later we should
do shallow learning, and model selection, as described in our doc.

TODO:
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
    * TODO: Include CUDA stuff here?
    * If disk speed is the limiting factor maybe we should train
    many models simultaneously with one disk read?
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class RandomProjectionPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, prediction_type: str):
        super().__init__()

        self.projection = torch.nn.Linear(nfeatures, nlabels)
        if prediction_type == "multilabel":
            self.activation = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        x = self.activation(x)
        return x


class SplitMemmapDataset(Dataset):
    """
    Embeddings are memmap'ed. (We don't use the labels yet, but will add them later.)

    WARNING: Don't shuffle this or access will be SLOW.
    """

    def __init__(self, embedding_path: Path, split_name: str):
        self.embedding_path = embedding_path
        self.split_name = split_name

        self.dim = tuple(
            json.load(
                open(embedding_path.joinpath(f"{split_name}.embedding-dimensions.json"))
            )
        )
        self.embedding_memmap = np.memmap(
            filename=embedding_path.joinpath(f"{split_name}.embeddings.npy"),
            dtype=np.float32,
            mode="r",
            shape=self.dim,
        )
        # self.labels = pickle.load(
        #     open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        # )
        # assert len(self.labels) == dim[0]

    def __len__(self):
        return self.dim[0]
        # return len(self.labels)

    def __getitem__(self, idx):
        return self.embedding_memmap[idx]
        # return self.embedding_memmap[idx], self.labels[idx]


def task_predictions(
    embedding_path: Path, scene_embedding_size: int, timestamp_embedding_size: int
):
    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())

    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1

    if metadata["embedding_type"] == "scene":
        predictor = RandomProjectionPrediction(
            scene_embedding_size, nlabels, metadata["prediction_type"]
        )
    elif metadata["embedding_type"] == "event":
        predictor = RandomProjectionPrediction(
            timestamp_embedding_size, nlabels, metadata["prediction_type"]
        )
    else:
        raise ValueError(f"embedding_type {metadata['embedding_type']} unknown.")

    # for split in metadata["splits"]:
    # Only do test for now, for the unlearned random predictor
    for split in [{"name": "test"}]:
        print(f"Getting embeddings for split: {split['name']}")

        dataloader = DataLoader(
            SplitMemmapDataset(embedding_path, split["name"]),
            batch_size=64,
            # We don't shuffle because it's slow.
            # Also we want predicted labels in the same order as
            # target labels.
            shuffle=False,
        )

        all_predicted_labels = []
        for embs in tqdm(dataloader):
            # for embs, target_labels in tqdm(dataloader):
            predicted_labels = predictor(embs)
            # TODO: Uses less memory to stack them one at a time
            all_predicted_labels.append(predicted_labels)
        all_predicted_labels = torch.cat(all_predicted_labels)
        pickle.dump(
            all_predicted_labels,
            open(
                embedding_path.joinpath(f"{split['name']}.predicted-labels.pkl"), "wb"
            ),
        )
