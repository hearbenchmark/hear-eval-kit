#!/usr/bin/env python3
"""
Map embeddings to predictions for every downstream task and store
test predictions to disk.

NOTE: Right now this is just a random projection. Later we should
do shallow learning, and model selection, as described in our doc.

TODO:
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
"""
import json
import os.path
from pathlib import Path
from importlib import import_module
from typing import Any, Dict, List, Tuple, Union

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TORCH = "torch"
TENSORFLOW = "tf"


class RandomProjectionPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, prediction_type: str):
        super().__init__()

        self.projection = torch.nn.Linear(self.nfeatures, self.nlabels)
        if prediction_type == "multilabel":
            self.activation = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.SoftMax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward(self, x: Tensor):
        x = self.projection(x)
        x = self.activation(x)
        return x


"""
class AudioFileDataset(Dataset):
    #Read in a JSON file and return audio and audio filenames

    def __init__(self, data: Dict, audio_dir: Path, sample_rate: int):
        self.filenames = list(data.keys())
        self.audio_dir = audio_dir
        assert self.audio_dir.is_dir()
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load in audio here in the Dataset. When the batch size is larger than
        # 1 then the torch dataloader can take advantage of multiprocessing.
        audio_path = self.audio_dir.joinpath(self.filenames[idx])
        audio, sr = sf.read(str(audio_path), dtype=np.float32)
        assert sr == self.sample_rate
        return audio, self.filenames[idx]


def get_dataloader_for_embedding(
    data_path: Path, audio_dir: Path, embedding: Embedding, batch_size: int = 64
):
    if embedding.type == TORCH:
        return DataLoader(
            AudioFileDataset(data_path, audio_dir, embedding.sample_rate),
            batch_size=batch_size,
            shuffle=True,
        )

    elif embedding.type == TENSORFLOW:
        raise NotImplementedError

    else:
        raise AssertionError("Unknown embedding type")
"""


def task_predictions(embedding_path: Path):
    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())

    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1

    ## TODO: Would be good to include the version here
    ## https://github.com/neuralaudio/hear2021-eval-kit/issues/37
    # embed_dir = Path("embeddings").joinpath(embedding.name)

    # for split in metadata["splits"]:
    # Only do test for now, for the unlearned random predictor
    for split in ["test"]:
        print(f"Getting embeddings for split: {split['name']}")

        split_path = embedding_path.joinpath(f"{split['name']}.json")
        assert split_path.is_file()

        # Root directory for audio files for this split
        audio_dir = embedding_path.joinpath(str(embedding.sample_rate), split["name"])

        # TODO: We might consider skipping files that already
        # have embeddings on disk, for speed
        # TODO: Choose batch size based upon audio file size?
        # Or assume that the API is smart enough to do this?
        # How do we test for memory blow up etc?
        # e.g. that it won't explode on 10 minute audio
        split_data = json.load(split_path.open())
        dataloader = get_dataloader_for_embedding(split_data, audio_dir, embedding, 4)

        outdir = embed_dir.joinpath(embedding_path.name, split["name"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for audios, filenames in tqdm(dataloader):
            labels = [split_data[file] for file in filenames]

            if metadata["task_type"] == "scene_labeling":
                embeddings = embedding.get_scene_embedding_as_numpy(audios)
                save_scene_embedding_and_label(embeddings, labels, filenames, outdir)

            elif metadata["task_type"] == "event_labeling":
                embeddings, timestamps = embedding.get_timestamp_embedding_as_numpy(
                    audios
                )
                labels = get_labels_for_timestamps(labels, timestamps, label_vocab)
                save_timestamp_embedding_and_label(
                    embeddings, timestamps, labels, filenames, outdir
                )

            else:
                raise ValueError(f"Unknown task type: {metadata['task_type']}")
