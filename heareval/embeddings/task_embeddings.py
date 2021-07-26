#!/usr/bin/env python3
"""
Compute the embeddings for every task and store to disk.

Since many tasks might be too large to store in GPU memory (or even
CPU memory), and because Wavenet-like models will be expensive at
inference time, we cache all embeddings to disk.

One benefit of this approach is that since all embeddings are cached
as numpy arrays, the final training code can be pytorch-only,
regardless of whether the embedding model is tensorflow based.

TODO:
    * Ideally, we would run this within a docker container, for
    security. https://github.com/neuralaudio/hear2021-eval-kit/issues/51
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
    * This is currently pytorch only.
    https://github.com/neuralaudio/hear2021-eval-kit/issues/52
    Using the included get_audio_embedding_numpy, we could instead
    have this work both for pytorch and tensorflow.
    https://github.com/neuralaudio/hear2021-eval-kit/issues/49
"""
import json
import os.path
from pathlib import Path
from importlib import import_module
from typing import Any, Tuple, Union

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TORCH = "torch"
TENSORFLOW = "tf"


class Embedding:
    def __init__(self, module_name: str, model_path: str = None):
        print(f"Importing {module_name}")
        self.module = import_module(module_name)

        # Load the model using the model weights path if they were provided
        if model_path is not None:
            print(f"Loading model using: {model_path}")
            self.model = self.module.load_model(model_path)
        else:
            self.model = self.module.load_model()

        # Check to see what type of model this is: torch or tensorflow
        if isinstance(self.model, torch.nn.Module):
            self.type = TORCH
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        elif isinstance(self.model, tf.Module):
            self.type = TENSORFLOW
            raise NotImplementedError("TensorFlow embeddings not supported yet.")
        else:
            raise TypeError(f"Unsupported model type received: {type(self.model)}")

    @property
    def name(self):
        return self.module.__name__

    @property
    def sample_rate(self):
        return self.model.sample_rate

    def as_tensor(self, x: Union[np.ndarray, torch.Tensor]):
        if self.type == TORCH:
            # Load array as tensor onto device
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, device=self.device)
            elif isinstance(x, torch.Tensor):
                x = x.to(self.device)
            else:
                raise TypeError(
                    "Input must be one of np.ndarray or torch.Tensor for"
                    f"torch audio embedding models. "
                    f"Received: {type(x)}"
                )

        elif self.type == TENSORFLOW:
            NotImplementedError("TensorFlow not implemented yet")
        else:
            raise AssertionError("Unknown type")

        return x

    def get_scene_embedding_as_numpy(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        audio = self.as_tensor(audio)
        if self.type == TORCH:
            embeddings = self.module.get_scene_embeddings(audio, self.model)
            return embeddings.detach().cpu().numpy()
        else:
            raise NotImplementedError("Not implemented for TF")

    def get_event_embedding_as_numpy(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        audio = self.as_tensor(audio)
        if self.type == TORCH:
            embeddings, timestamps = self.module.get_event_embeddings(audio, self.model)
            embeddings = embeddings.detach().cpu().numpy()
            timestamps = timestamps.detach().cpu().numpy()
            return embeddings, timestamps
        else:
            raise NotImplementedError("Not implemented for TF")


class AudioFileDataset(Dataset):
    """
    Read in a JSON file and return audio filenames and labels
    """

    def __init__(self, json_file: Path, audio_dir: Path, sample_rate: int):
        self.data = json.load(json_file.open())
        self.keys = list(self.data.keys())
        self.audio_dir = audio_dir
        assert self.audio_dir.is_dir()
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load in audio here in the Dataset. When the batch size is larger than
        # 1 then the torch dataloader can take advantage of multiprocessing.
        audio_path = self.audio_dir.joinpath(self.keys[idx])
        audio, sr = sf.read(audio_path, dtype=np.float32)
        assert sr == self.sample_rate
        return audio, self.data[self.keys[idx]]["label"], self.keys[idx]


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


def save_scene_embedding_and_label(
    embeddings: np.ndarray, labels: Any, filename: Tuple, outdir: Path
):
    for i, file in enumerate(filename):
        out_file = outdir.joinpath(f"{file}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        np.save(f"{out_file}.label.npy", labels[i])


def task_embeddings(embedding: Embedding, task_path: Path):

    metadata = json.load(task_path.joinpath("task_metadata.json").open())
    task_type = metadata["task_type"]

    # TODO: Would be good to include the version here
    # https://github.com/neuralaudio/hear2021-eval-kit/issues/37
    embed_dir = Path("embeddings").joinpath(embedding.name)

    for split in metadata["splits"]:
        print(f"Getting embeddings for split: {split['name']}")

        split_path = task_path.joinpath(f"{split['name']}.json")
        assert split_path.is_file()

        # Root directory for audio files for this split
        audio_dir = task_path.joinpath(str(embedding.sample_rate), split["name"])

        # TODO: We might consider skipping files that already
        # have embeddings on disk, for speed
        # TODO: Choose batch size based upon audio file size?
        # Or assume that the API is smart enough to do this?
        # How do we test for memory blow up etc?
        # e.g. that it won't explode on 10 minute audio
        dataloader = get_dataloader_for_embedding(split_path, audio_dir, embedding, 4)

        outdir = embed_dir.joinpath(task_path.name, split["name"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for audio, label, filename in tqdm(dataloader):
            if metadata["task_type"] == "scene_labeling":
                embeddings = embedding.get_scene_embedding_as_numpy(audio)
                save_scene_embedding_and_label(embeddings, label, filename, outdir)

            elif metadata["task_type"] == "event_labeling":
                embeddings, timestamps = embedding.get_event_embedding_as_numpy(audio)

            else:
                raise ValueError(f"Unknown task type: {metadata['task_type']}")
