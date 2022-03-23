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
import pickle
import random
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
from intervaltree import IntervalTree
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# import wandb
import heareval.gpu_max_mem as gpu_max_mem

TORCH = "torch"
TENSORFLOW = "tf"


class Embedding:
    """
    A wrapper class to help with loading embedding models and computing embeddings
    using the HEAR 2021 API for both torch and tensorflow models.
    # TODO: Still need to implement and test this with TensorFlow

    Args:
        module_name: the import name for the embedding module
        model_path: location to load the model from
    """

    def __init__(
        self,
        module_name: str,
        model_path: str = None,
        model_options: Optional[Dict[str, Any]] = None,
    ):
        print(f"Importing {module_name}")
        self.module = import_module(module_name)

        if model_options is None:
            model_options = {}

        # Load the model using the model weights path if they were provided
        if model_path is not None:
            print(f"Loading model using: {model_path}")
            self.model = self.module.load_model(model_path, **model_options)  # type: ignore
        else:
            self.model = self.module.load_model(**model_options)  # type: ignore

        # Check to see what type of model this is: torch or tensorflow
        if isinstance(self.model, torch.nn.Module):
            self.type = TORCH
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        elif isinstance(self.model, tf.Module):
            self.type = TENSORFLOW
            # Tensorflow automatically manages data transfers to device,
            # so we don't need to set self.device
        else:
            raise TypeError(f"Unsupported model type received: {type(self.model)}")

    @property
    def name(self):
        # TODO: would be nice to include version in this string, a versioned string.
        #   Potentially can set a version from the command line too to help with testing
        #   the same model but with difference versions of the weights.
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
            # Load array as tensor onto device

            if not isinstance(x, np.ndarray):
                x = x.numpy()
            x = tf.convert_to_tensor(x)
        else:
            raise AssertionError("Unknown type")

        return x

    def get_scene_embedding_as_numpy(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        audio = self.as_tensor(audio)
        if self.type == TORCH:
            with torch.no_grad():
                embeddings = self.module.get_scene_embeddings(  # type: ignore
                    audio, self.model
                )
                return embeddings.detach().cpu().numpy()
        elif self.type == TENSORFLOW:
            embeddings = self.module.get_scene_embeddings(  # type: ignore
                audio, self.model
            )
            return embeddings.numpy()
        else:
            raise NotImplementedError("Unknown type")

    def get_timestamp_embedding_as_numpy(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        audio = self.as_tensor(audio)
        if self.type == TORCH:
            with torch.no_grad():
                # flake8: noqa
                embeddings, timestamps = self.module.get_timestamp_embeddings(  # type: ignore
                    audio,
                    self.model,
                )
                gpu_max_mem.measure()
                embeddings = embeddings.detach().cpu().numpy()
                timestamps = timestamps.detach().cpu().numpy()
                return embeddings, timestamps
        elif self.type == TENSORFLOW:
            # flake8: noqa
            embeddings, timestamps = self.module.get_timestamp_embeddings(  # type: ignore
                audio,
                self.model,
            )
            gpu_max_mem.measure()
            embeddings = embeddings.numpy()
            timestamps = timestamps.numpy()
            return embeddings, timestamps
        else:
            raise NotImplementedError("Unknown type")


class AudioFileDataset(Dataset):
    """
    Read in a JSON file and return audio and audio filenames
    """

    def __init__(self, data: Dict, audio_dir: Path, sample_rate: int):
        self.filenames = list(data.keys())
        self.audio_dir = audio_dir
        assert self.audio_dir.is_dir(), f"{audio_dir} is not a directory"
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
    data: Dict, audio_dir: Path, embedding: Embedding, batch_size: int = 64
):
    if embedding.type == TORCH or embedding.type == TENSORFLOW:
        return DataLoader(
            AudioFileDataset(data, audio_dir, embedding.sample_rate),
            batch_size=batch_size,
            shuffle=False,
        )

    else:
        raise AssertionError("Unknown embedding type")


def save_scene_embedding_and_labels(
    embeddings: np.ndarray, labels: List[Dict], filenames: Tuple[str], outdir: Path
):
    assert len(embeddings) == len(filenames)
    assert len(labels) == len(filenames)
    for i, filename in enumerate(filenames):
        out_file = outdir.joinpath(f"{filename}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        json.dump(labels[i], open(f"{out_file}.target-labels.json", "w"))


def save_timestamp_embedding_and_labels(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    filename: Tuple[str],
    outdir: Path,
):
    for i, file in enumerate(filename):
        out_file = outdir.joinpath(f"{file}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        assert len(timestamps[i].shape) == 1
        json.dump(timestamps[i].tolist(), open(f"{out_file}.timestamps.json", "w"))
        json.dump(labels[i], open(f"{out_file}.target-labels.json", "w"), indent=4)


def get_labels_for_timestamps(labels: List, timestamps: np.ndarray) -> List:
    # -> List[List[List[str]]]:
    # -> List[List[str]]:
    # TODO: Is this function redundant?
    # A list of labels present at each timestamp
    timestamp_labels = []

    # NOTE: Make sure dataset events are specified in ms.
    assert len(labels) == len(timestamps)
    for i, label in enumerate(labels):
        tree = IntervalTree()
        # Add all events to the label tree
        for event in label:
            # We add 0.0001 so that the end also includes the event
            tree.addi(event["start"], event["end"] + 0.0001, event["label"])

        labels_for_sound = []
        # Update the binary vector of labels with intervals for each timestamp
        for j, t in enumerate(timestamps[i]):
            interval_labels: List[str] = [interval.data for interval in tree[t]]
            labels_for_sound.append(interval_labels)
            # If we want to store the timestamp too
            # labels_for_sound.append([float(t), interval_labels])

        timestamp_labels.append(labels_for_sound)

    assert len(timestamp_labels) == len(timestamps)
    return timestamp_labels


def memmap_embeddings(
    outdir: Path,
    prng: random.Random,
    metadata: Dict,
    split_name: str,
    embed_task_dir: Path,
    split_data: Dict,
):
    """
    Memmap all the embeddings to one file, and pickle all the labels.
    (We assume labels can fit in memory.)
    TODO: This writes things to disk double, we could clean that up after.
    We might also be able to get away with writing to disk only once.
    """
    embedding_files = [outdir.joinpath(f"{f}.embedding.npy") for f in split_data.keys()]
    prng.shuffle(embedding_files)

    # First count the number of embeddings total
    nembeddings = 0
    ndim: int
    for embedding_file in tqdm(embedding_files):
        assert embedding_file.exists()
        emb = np.load(embedding_file).astype(np.float32)
        if metadata["embedding_type"] == "scene":
            assert emb.ndim == 1
            nembeddings += 1
            ndim = emb.shape[0]
            assert emb.dtype == np.float32
        elif metadata["embedding_type"] == "event":
            assert emb.ndim == 2
            nembeddings += emb.shape[0]
            ndim = emb.shape[1]
            assert emb.dtype == np.float32
        else:
            raise ValueError(f"Unknown embedding type: {metadata['embedding_type']}")

    open(
        embed_task_dir.joinpath(f"{split_name}.embedding-dimensions.json"), "wt"
    ).write(json.dumps((nembeddings, ndim)))
    embedding_memmap = np.memmap(
        filename=embed_task_dir.joinpath(f"{split_name}.embeddings.npy"),
        dtype=np.float32,
        mode="w+",
        shape=(nembeddings, ndim),
    )
    idx = 0
    labels = []
    filename_timestamps = []
    for embedding_file in tqdm(embedding_files):
        emb = np.load(embedding_file)
        lbl = json.load(
            open(str(embedding_file).replace("embedding.npy", "target-labels.json"))
        )

        if metadata["embedding_type"] == "scene":
            assert emb.ndim == 1
            embedding_memmap[idx] = emb
            # lbl will be a list of labels, make sure that it has exactly one label
            # for multiclass problems. Will be a list of zero or more for multilabel.
            if metadata["prediction_type"] == "multiclass":
                assert len(lbl) == 1
            elif metadata["prediction_type"] == "multilabel":
                assert isinstance(lbl, list)
            else:
                NotImplementedError(
                    "Only multiclass and multilabel prediction types"
                    f"implemented for scene embeddings. Received {metadata['prediction_type']}"
                )

            labels.append(lbl)
            idx += 1
        elif metadata["embedding_type"] == "event":
            assert emb.ndim == 2
            embedding_memmap[idx : idx + emb.shape[0]] = emb
            assert emb.shape[0] == len(lbl)
            labels += lbl

            timestamps = json.load(
                open(str(embedding_file).replace("embedding.npy", "timestamps.json"))
            )
            slug = str(embedding_file).replace(".embedding.npy", "")
            filename_timestamps += [(slug, timestamp) for timestamp in timestamps]
            assert emb.shape[0] == len(
                timestamps
            ), f"{emb.shape[0]} != {len(timestamps)}"
            assert len(lbl) == len(timestamps), f"{len(lbl)} != {len(timestamps)}"

            idx += emb.shape[0]
        else:
            raise ValueError(f"Unknown embedding type: {metadata['embedding_type']}")

    # Write changes to disk
    embedding_memmap.flush()
    # TODO: Convert labels to indices?
    pickle.dump(
        labels,
        open(
            embed_task_dir.joinpath(f"{split_name}.target-labels.pkl"),
            "wb",
        ),
    )
    if metadata["embedding_type"] == "event":
        assert len(labels) == len(filename_timestamps)
        open(
            embed_task_dir.joinpath(f"{split_name}.filename-timestamps.json"),
            "wt",
        ).write(json.dumps(filename_timestamps, indent=4))


def task_embeddings(
    embedding: Embedding,
    task_path: Path,
    embed_task_dir: Path,
):
    prng = random.Random()
    prng.seed(0)

    metadata_path = task_path.joinpath("task_metadata.json")
    metadata = json.load(metadata_path.open())
    label_vocab_path = task_path.joinpath("labelvocabulary.csv")

    # wandb.init(project="heareval", tags=["embedding", task_name])

    # Copy these two files to the embeddings directory,
    # so we have everything we need in embeddings for doing downstream
    # prediction and evaluation.
    if not os.path.exists(embed_task_dir):
        os.makedirs(embed_task_dir)
    shutil.copy(metadata_path, embed_task_dir)
    shutil.copy(label_vocab_path, embed_task_dir)

    for split in metadata["splits"]:
        print(f"Getting embeddings for split: {split}")

        split_path = task_path.joinpath(f"{split}.json")
        assert split_path.is_file()

        # Copy over the ground truth labels as they may be needed for evaluation
        shutil.copy(split_path, embed_task_dir)

        # Root directory for audio files for this split
        audio_dir = task_path.joinpath(str(embedding.sample_rate), split)

        # TODO: We might consider skipping files that already
        # have embeddings on disk, for speed.
        # This was based upon futzing with various models
        # on the dcase task.
        # Unforunately, this is not tuned per model and is based upon the largest
        # model and largest audio files we have.
        estimated_batch_size: int
        if metadata["sample_duration"] is not None:
            estimated_batch_size = max(
                1,
                int(
                    # 0.9
                    # One of the submissions needs smaller batches
                    0.7
                    * (120 / metadata["sample_duration"])
                    * (16000 / embedding.sample_rate)
                ),
            )
        else:
            # If the sample duration is None, we use a batch size of 1 as the audio
            # files will of different length and the model cannot be run with
            # batch size > 1
            estimated_batch_size = 1
        print(f"Estimated batch size = {estimated_batch_size}")
        split_data = json.load(split_path.open())
        dataloader = get_dataloader_for_embedding(
            split_data, audio_dir, embedding, batch_size=estimated_batch_size
        )

        outdir = embed_task_dir.joinpath(split)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for audios, filenames in tqdm(dataloader):
            labels = [split_data[file] for file in filenames]

            if metadata["embedding_type"] == "scene":
                embeddings = embedding.get_scene_embedding_as_numpy(audios)
                save_scene_embedding_and_labels(embeddings, labels, filenames, outdir)

            elif metadata["embedding_type"] == "event":
                embeddings, timestamps = embedding.get_timestamp_embedding_as_numpy(
                    audios
                )
                labels = get_labels_for_timestamps(labels, timestamps)
                assert len(labels) == len(filenames)
                assert len(labels[0]) == len(timestamps[0])
                save_timestamp_embedding_and_labels(
                    embeddings, timestamps, labels, filenames, outdir
                )

            else:
                raise ValueError(
                    f"Unknown embedding type: {metadata['embedding_type']}"
                )

        memmap_embeddings(outdir, prng, metadata, split, embed_task_dir, split_data)
