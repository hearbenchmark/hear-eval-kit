#!/usr/bin/env python3
"""
Compute the embeddings for every task.

TODO:
    * Ideally, we would run this within a docker container, for
    security.
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
    * This is currently pytorch only. If we instead had a method
    get_audio_embedding_numpy, we could instead have this work both
    for pytorch and tensorflow.
    https://github.com/neuralaudio/hear2021-eval-kit/issues/49
"""

import csv
import glob
import os.path
import pickle
from importlib import import_module
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# This could instead be something from the participants
EMBEDDING_PIP = "heareval.baseline"
EMBEDDING_MODEL_PATH = ""  # Not used by baseline

# TODO: Support for multiple GPUs?
device = "gpu" if torch.cuda.is_available() else "cpu"

EMBED = import_module(EMBEDDING_PIP)


class CSVDataset(Dataset):
    """
    Read in a CSV file, and return data rows as [column 1, (column 2, ...)]
    """

    def __init__(self, csv_file):
        # Our CSV files don't have headers, so we can't use
        # pd.read_csv
        # I'm on the fence whether we want CSV headers or not,
        # since the format is standardized.
        with open(csv_file) as f:
            csvreader = csv.reader(f)
            self.rows = [row for row in csvreader]
        ncol = len(self.rows[0])
        # Make sure all rows have the same number of column,
        # and rewrite as [col1, (col2, ...)]
        for idx, row in enumerate(self.rows):
            assert len(row) == ncol
            self.rows[idx] = [row[0], row[1:]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# These is a high-level wrappers on the basic API that we might
# consider sharing among all participants.


def get_audio_embedding_numpy(
    audio_numpy: np.ndarray,
    model: Any,
    frame_rate: float,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    # Is there a way to avoid this .float() cast? Or do it on the
    # numpy, not after the tensor creation?
    embedding_dict, timestamps = EMBED.get_audio_embedding(
        torch.tensor(audio_numpy, device=device).float(),
        model=model,
        frame_rate=frame_rate,
    )
    for key in embedding_dict.keys():
        embedding_dict[key] = embedding_dict[key].detach().cpu().numpy()
    timestamps = timestamps.detach().cpu().numpy()
    return embedding_dict, timestamps


if __name__ == "__main__":
    model = EMBED.load_model(EMBEDDING_MODEL_PATH, device=device)

    # TODO: Would be good to include the version here
    embeddir = os.path.join("embeddings", EMBED.__name__)

    for task in glob.glob("tasks/*"):
        # TODO: We should be reading the metadata that describes
        # the frame_rate.
        frame_rate = 10

        # TODO: Include "val" ?
        for split in ["train", "test"]:
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
                        os.path.join(task, str(EMBED.input_sample_rate()), split, f)
                    )
                    assert sr == EMBED.input_sample_rate()
                    audios.append(x)
                audios = np.vstack(audios)
                embedding_dict, timestamps = get_audio_embedding_numpy(
                    audios, model=model, frame_rate=frame_rate
                )
                for i, filename in enumerate(files):
                    file_embedding_dict = {
                        emb_size: embedding_dict[emb_size][i]
                        for emb_size in embedding_dict
                    }
                    pickle.dump(
                        file_embedding_dict,
                        open(os.path.join(outdir, filename + ".pkl"), "wb"),
                    )
