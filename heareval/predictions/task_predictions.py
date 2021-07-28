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

from collections import OrderedDict
import json
import pickle
from pathlib import Path
from typing import Dict, List

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class RandomProjectionPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, prediction_type: str):
        super().__init__()

        self.projection = torch.nn.Linear(nfeatures, nlabels)
        torch.nn.init.normal_(self.projection.weight)
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


def create_events_from_prediction(
    predictions: Dict, threshold: float = 0.5
) -> IntervalTree:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events.

    Args:
        predictions: A dictionary of predictions {timestamp -> prediction}
        threshold: Threshold for determining whether to apply a label

    Returns:
        An IntervalTree object containing all the events from the predictions.
    """
    # Make sure the timestamps are in the correct order
    timestamps = sorted(predictions.keys())

    # Create a sorted tensor of frame level predictions for this file.
    predictions = torch.stack([predictions[t] for t in timestamps])

    # Apply thresholding to get the label detected at each frame.
    # TODO: Apply median filtering for smoothing?
    #   Additionally, DCASE limits the number of active classes at a single
    #   time to 6. We could do that here as well.
    predictions = (predictions > threshold).type(torch.int8)

    # Difference between each timestamp to find event boundaries
    pred_diff = torch.diff(predictions, dim=0)

    # This is slow, but for each class look through all the timestamp predictions
    # and construct a set of events with onset and offset timestamps.
    event_tree = IntervalTree()
    for class_idx in range(predictions.shape[1]):
        # Check to see if this starts with an active event
        current_event = None
        if predictions[0][class_idx] == 1:
            assert pred_diff[0][class_idx] != 1
            current_event = [timestamps[0]]

        for t in range(pred_diff.shape[0]):
            # New onset
            if pred_diff[t][class_idx] == 1:
                assert current_event is None
                current_event = [timestamps[t]]

            # Offset for current event
            elif pred_diff[t][class_idx] == -1:
                assert len(current_event) == 1
                # TODO: filter on event length? DCASE only included events that
                #   are 60ms long.
                event_tree.addi(
                    begin=current_event[0], end=timestamps[t + 1], data=class_idx
                )
                current_event = None

    return event_tree


def get_predictions_as_events(
    predictions: torch.Tensor, file_timestamps: List, label_vocab: pd.DataFrame
) -> Dict[str, List]:
    """
    Produces lists of events from a set of frame based label probabilities. The input
    prediction tensor may contain frame predictions from a set of different files
    concatenated together. We want to compute events on each of those files separately.

    Args:
        predictions: a tensor of frame based multi-label predictions.
        file_timestamps: a list of filenames and timestamps where each entry corresponds
            to a frame in the predictions tensor.
        label_vocab: The set of labels and their associated int idx

    Returns:
        A dictionary of lists of events keyed on the filename slug
    """
    # This probably could be more efficient if we make the assumption that
    # timestamps are in sorted order. But this makes sure of it.
    assert predictions.shape[0] == len(file_timestamps)
    event_files = {}
    for i, file_timestamp in enumerate(file_timestamps):
        filename, timestamp = file_timestamp
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in event_files:
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][timestamp] = predictions[i]

    # Dictionary of labels: {idx -> label}
    label_dict = label_vocab.set_index("idx").to_dict()["label"]

    # Create events for all the different files. Store all the events as a dictionary
    # with the same format as the ground truth from the luigi pipeline.
    # Ex) { slug -> [{"label" : "woof", "start": 0.0, "end": 2.32}, ...], ...}
    event_dict = {}
    for slug, timestamp_predictions in tqdm(event_files.items()):
        event_tree = create_events_from_prediction(timestamp_predictions)
        events = []
        for interval in sorted(event_tree):
            label = label_dict[interval.data]
            # TODO: start and end? Let's use begin and end like interval tree?
            events.append(
                {"label": label, "start": interval.begin, "end": interval.end}
            )

        event_dict[slug] = events

    return event_dict


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

        if metadata["embedding_type"] == "event":
            # For event predictions we need to convert the frame-based predictions
            # to a list of events with start and stop timestamps. These events are
            # computed on each file independently and then saved as JSON in the same
            # format as the ground truth events produced by the luigi pipeline.

            # A list of filenames and timestamps associated with each prediction
            file_timestamps = json.load(
                embedding_path.joinpath(
                    f"{split['name']}.filename-timestamps.json"
                ).open()
            )

            print("Creating events from predictions:")
            events = get_predictions_as_events(
                all_predicted_labels, file_timestamps, label_vocab
            )

            json.dump(
                events,
                embedding_path.joinpath(f"{split['name']}.predictions.json").open("w"),
                indent=4,
            )

        pickle.dump(
            all_predicted_labels,
            open(
                embedding_path.joinpath(f"{split['name']}.predicted-labels.pkl"), "wb"
            ),
        )
