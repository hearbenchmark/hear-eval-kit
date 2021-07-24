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
import json
import os.path
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# This could instead be something from the participants
# TODO: Command line or config?
EMBEDDING_PIP = "hearbaseline"
EMBEDDING_MODEL_PATH = ""  # Baseline doesn't load model
# TODO: Actually load baseline model

# TODO: Support for multiple GPUs?
device = "cuda" if torch.cuda.is_available() else "cpu"

EMBED = import_module(EMBEDDING_PIP)


class AudioFileDataset(Dataset):
    """
    Read in a CSV file, and return audio filenames.
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


# High-level wrapper on the basic API that we might
# consider sharing among all participants.
def get_audio_embedding_numpy(
    audio_numpy: np.ndarray,
    model: Any,
    task_type: str,
) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
    if task_type == "scene_labeling":
        embeddings = EMBED.get_scene_embeddings(  # type: ignore
            torch.tensor(audio_numpy, device=device),
            model=model,
        )
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings, None
    elif task_type == "event_labeling":
        embeddings, timestamps = EMBED.get_timestamp_embeddings(  # type: ignore
            torch.tensor(audio_numpy, device=device),
            model=model,
        )
        embeddings = embeddings.detach().cpu().numpy()
        timestamps = timestamps.detach().cpu().numpy()
        # TODO: Turn this into a validator test
        assert len(embeddings) == len(timestamps)
        return embeddings, timestamps
    else:
        raise ValueError(f"Unknown task_type = {task_type}")


def task_embeddings():
    model = EMBED.load_model(EMBEDDING_MODEL_PATH)

    # TODO: Would be good to include the version here
    # https://github.com/neuralaudio/hear2021-eval-kit/issues/37
    embeddir = os.path.join("embeddings", EMBED.__name__)  # type: ignore
    embed_sr = model.sample_rate

    # TODO: tqdm with task name
    for task in glob.glob("tasks/*"):
        print(f"Task {task}")
        task_config = json.loads(open(os.path.join(task, "task_metadata.json")).read())
        for split in ["train", "valid", "test"]:
            print(f"Getting embeddings for {split} split:")

            metadata_csv = os.path.join(task, f"{split}.csv")
            if not os.path.exists(metadata_csv):
                continue

            # TODO: We might consider skipping files that already
            # have embeddings on disk, for speed
            # TODO: Choose batch size based upon audio file size?
            # Or assume that the API is smart enough to do this?
            # How do we test for memory blow up etc?
            # e.g. that it won't explode on 10 minute audio
            dataloader = DataLoader(
                AudioFileDataset(metadata_csv),
                batch_size=1,  # batch_size=64,
                shuffle=True,
            )
            outdir = os.path.join(embeddir, task, split)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for batch in tqdm(dataloader):
                files = batch
                audios = []
                for f in files:
                    x, sr = sf.read(
                        os.path.join(task, str(embed_sr), split, f), dtype=np.float32
                    )
                    assert sr == embed_sr
                    audios.append(x)

                audios = np.vstack(audios)
                task_type = task_config["task_type"]
                embeddings, timestamps = get_audio_embedding_numpy(
                    audios, model=model, task_type=task_type
                )
                assert len(files) == embeddings.shape[0]
                for i, filename in enumerate(files):
                    if timestamps is not None:
                        assert task_type in ["event_labeling"]
                        np.save(
                            os.path.join(outdir, f"{filename}.npy"),
                            (embeddings[i], timestamps[i]),
                        )
                    else:
                        assert task_type in ["scene_labeling"]
                        np.save(os.path.join(outdir, f"{filename}.npy"), embeddings[i])


if __name__ == "__main__":
    task_embeddings()
