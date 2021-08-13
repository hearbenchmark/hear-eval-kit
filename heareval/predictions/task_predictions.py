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

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

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
            self.activation: torch.nn.Module = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward(self, x: torch.Tensor):
        x, y = x
        x = self.projection(x)
        x = self.activation(x)
        return x


class SplitMemmapDataset(Dataset):
    """
    Embeddings are memmap'ed. (We don't use the labels yet, but will add them later.)

    WARNING: Don't shuffle this or access will be SLOW.
    """

    def __init__(
        self, embedding_path: Path, prediction_type: str, nlabels: int, split_name: str
    ):
        self.embedding_path = embedding_path
        self.prediction_type = prediction_type
        self.nlabels = nlabels
        self.split_name = split_name

        self.label_to_idx = {
            label: int(idx)
            for idx, label in csv.reader(
                embedding_path.joinpath("labelvocabulary.csv").open()
            )
            if (idx, label) != ("idx", "label")
        }

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
        self.labels = pickle.load(
            open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        )
        assert len(self.labels) == self.dim[0]
        assert (
            self.embedding_memmap[0].shape[0] == self.dim[1]
        ), f"{self.embedding_memmap[0].shape[0]}, {self.dim[1]}"

    def __len__(self) -> int:
        return self.dim[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding_memmap[idx]
        y = [self.label_to_idx[str(label)] for label in self.labels[idx]]
        if self.prediction_type == "multilabel":
            # Lame special case
            if not y:
                return x, torch.zeros((self.nlabels,), dtype=torch.int64)
            return (
                x,
                torch.nn.functional.one_hot(
                    torch.LongTensor(y), num_classes=self.nlabels
                )
                .max(axis=0)
                .values,
            )
        elif self.prediction_type == "multiclass":
            assert len(y) == 1
            return x, y[0]
        else:
            raise ValueError(f"Unknown prediction_type {self.prediction_type}")


def create_events_from_prediction(
    prediction_dict: Dict[float, torch.Tensor],
    threshold: float = 0.5,
    min_duration=60.0,
) -> IntervalTree:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events. This
    converts the prediction tensor to a binary label based on the threshold value. Any
    events occurring at adjacent timestamps are considered to be part of the same event.
    This loops through and creates events for each label class. Disregards events that
    are less than the min_duration milliseconds.

    Args:
        prediction_dict: A dictionary of predictions keyed on timestamp
            {timestamp -> prediction}. The prediction is a tensor of label
            probabilities.
        threshold: Threshold for determining whether to apply a label
        min_duration: the minimum duration in milliseconds for an
                event to be included.

    Returns:
        An IntervalTree object containing all the events from the predictions.
    """
    # Make sure the timestamps are in the correct order
    timestamps = np.array(sorted(prediction_dict.keys()))

    # Create a sorted numpy matrix of frame level predictions for this file. We convert
    # to a numpy array here before applying a median filter.
    predictions = np.stack([prediction_dict[t].detach().numpy() for t in timestamps])

    # We can apply a median filter here to smooth out events, but b/c participants
    # can select their own timestamp interval, selecting the filter window becomes
    # challenging and could give an unfair advantage -- so we leave out for now.
    # This could look something like this:
    # ts_diff = timestamps[1] - timestamps[0]
    # filter_width = int(round(median_filter_ms / ts_diff))
    # predictions = median_filter(predictions, size=(filter_width, 1))

    # Convert probabilities to binary vectors based on threshold
    predictions = (predictions > threshold).astype(np.int8)

    # Difference between each timestamp to find event boundaries
    pred_diff = np.diff(predictions, axis=0)

    # This is slow, but for each class look through all the timestamp predictions
    # and construct a set of events with onset and offset timestamps.
    event_tree = IntervalTree()
    for class_idx in range(predictions.shape[1]):
        # Check to see if this starts with an active event
        current_event = []
        if predictions[0][class_idx] == 1:
            assert pred_diff[0][class_idx] != 1
            current_event = [timestamps[0]]

        for t in range(pred_diff.shape[0]):
            # New onset
            if pred_diff[t][class_idx] == 1:
                assert len(current_event) == 0
                current_event = [timestamps[t]]

            # Offset for current event
            elif pred_diff[t][class_idx] == -1:
                assert len(current_event) == 1
                current_event.append(timestamps[t + 1])
                event_duration = current_event[1] - current_event[0]

                # Add event if greater than the minimum duration threshold
                if event_duration >= min_duration:
                    event_tree.addi(
                        begin=current_event[0], end=current_event[1], data=class_idx
                    )
                current_event = []

    return event_tree


def get_events_for_all_files(
    predictions: torch.Tensor, file_timestamps: List, label_vocab: pd.DataFrame
) -> Dict[str, List]:
    """
    Produces lists of events from a set of frame based label probabilities.
    The input prediction tensor may contain frame predictions from a set of different
    files concatenated together. file_timestamps has a list of filenames and
    timestamps for each frame in the predictions tensor.

    We split the predictions into separate tensors based on the filename and compute
    events based on those individually.

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
    event_files: Dict[str, Dict[float, torch.Tensor]] = {}
    for i, file_timestamp in enumerate(file_timestamps):
        filename, timestamp = file_timestamp
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in event_files:
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][float(timestamp)] = predictions[i]

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


def label_vocab_nlabels(embedding_path: Path) -> Tuple[pd.DataFrame, int]:
    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    return (label_vocab, nlabels)


def task_predictions_train(
    embedding_path: Path, scene_embedding_size: int, timestamp_embedding_size: int
) -> torch.nn.Module:
    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocal, nlabels = label_vocab_nlabels(embedding_path)

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
    return predictor


def task_predictions_test(
    predictor: torch.nn.Module,
    embedding_path: Path,
    scene_embedding_size: int,
    timestamp_embedding_size: int,
):
    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)

    if metadata["embedding_type"] == "scene":
        predictor = RandomProjectionPrediction(
            scene_embedding_size, nlabels, metadata["prediction_type"]
        )
    elif metadata["embedding_type"] == "event":
        predictor = RandomProjectionPrediction(
            timestamp_embedding_size, nlabels, metadata["prediction_type"]
        )

    # for split in metadata["splits"]:
    for split in [{"name": "test"}]:
        print(f"Getting embeddings for split: {split['name']}")

        dataloader = DataLoader(
            SplitMemmapDataset(
                embedding_path, metadata["prediction_type"], nlabels, split["name"]
            ),
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
        predicted_labels = torch.cat(all_predicted_labels)

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
            events = get_events_for_all_files(
                predicted_labels, file_timestamps, label_vocab
            )

            json.dump(
                events,
                embedding_path.joinpath(f"{split['name']}.predictions.json").open("w"),
                indent=4,
            )

        pickle.dump(
            predicted_labels,
            open(
                embedding_path.joinpath(f"{split['name']}.predicted-labels.pkl"), "wb"
            ),
        )


def task_predictions(
    embedding_path: Path, scene_embedding_size: int, timestamp_embedding_size: int
):
    predictor = task_predictions_train(
        embedding_path, scene_embedding_size, timestamp_embedding_size
    )
    task_predictions_test(
        predictor, embedding_path, scene_embedding_size, timestamp_embedding_size
    )
