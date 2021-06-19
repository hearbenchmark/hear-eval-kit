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

import glob
import os.path
from importlib import import_module
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from csvdataset import CSVDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# This could instead be something from the participants
EMBEDDING_PIP = "heareval.model.baseline"
EMBEDDING_MODEL_PATH = ""  # Baseline doesn't load model

# TODO: Support for multiple GPUs?
device = "cuda" if torch.cuda.is_available() else "cpu"

EMBED = import_module(EMBEDDING_PIP)


def get_audio_embedding_numpy(
    audio_numpy: np.ndarray,
    model: Any,
    hop_size: float,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    embedding, timestamps = EMBED.get_audio_embedding(  # type: ignore
        torch.tensor(audio_numpy, device=device),
        model=model,
        hop_size=hop_size,
    )
    embedding = embedding.detach().cpu().numpy()
    timestamps = timestamps.detach().cpu().numpy()
    return embedding, timestamps


def task_embeddings():
    model = EMBED.load_model(EMBEDDING_MODEL_PATH, device=device)  # type: ignore

    # TODO: Would be good to include the version here
    # https://github.com/neuralaudio/hear2021-eval-kit/issues/37
    embeddir = os.path.join("embeddings", EMBED.__name__)  # type: ignore
    embed_sr = model.sample_rate

    for task in glob.glob("tasks/*"):
        # TODO: We should be reading the metadata that describes
        # the hop_size.
        # https://github.com/neuralaudio/hear2021-eval-kit/issues/53
        hop_size = 0.1

        # TODO: Include "val" ?
        for split in ["train", "test"]:
            print(f"Getting embeddings for {split} split:")

            # TODO: We might consider skipping files that already
            # have embeddings on disk, for speed
            dataloader = DataLoader(
                CSVDataset(os.path.join(task, f"{split}.csv")),
                batch_size=64,
                shuffle=True,
            )
            outdir = os.path.join(embeddir, task, split)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for batch in tqdm(dataloader):
                files, labels = batch
                audios = []
                for f in files:
                    x, sr = sf.read(
                        os.path.join(task, str(embed_sr), split, f), dtype=np.float32
                    )
                    assert sr == embed_sr
                    audios.append(x)

                audios = np.vstack(audios)
                embedding, timestamps = get_audio_embedding_numpy(
                    audios, model=model, hop_size=hop_size
                )

                for i, filename in enumerate(files):
                    np.save(os.path.join(outdir, f"{filename}.npy"), embedding[i])


if __name__ == "__main__":
    task_embeddings()
