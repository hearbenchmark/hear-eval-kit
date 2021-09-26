#!/usr/bin/env python3
"""
Map embeddings to predictions for every downstream task and store
test predictions to disk.

Model selection over the validation score.

TODO:
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
    * If disk speed is the limiting factor maybe we should train
    many models simultaneously with one disk read?
"""

import copy
import json
import math
import multiprocessing
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import more_itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchinfo

# import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.ndimage import median_filter
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from heareval.score import (
    ScoreFunction,
    available_scores,
    label_to_binary_vector,
    label_vocab_as_dict,
)

TASK_SPECIFIC_PARAM_GRID = {
    "dcase2016_task2": {
        # sed_eval is very slow
        "check_val_every_n_epoch": [10],
    }
}

PARAM_GRID = {
    "hidden_layers": [1, 2],
    # "hidden_layers": [0, 1, 2],
    # "hidden_layers": [1, 2, 3],
    "hidden_dim": [1024],
    # "hidden_dim": [256, 512, 1024],
    # "hidden_dim": [1024, 512],
    # Encourage 0.5
    "dropout": [0.1],
    # "dropout": [0.1, 0.5],
    # "dropout": [0.1, 0.3],
    # "dropout": [0.1, 0.3, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5, 1e-5],
    # "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "patience": [20],
    "max_epochs": [500],
    # "max_epochs": [500, 1000],
    "check_val_every_n_epoch": [3],
    # "check_val_every_n_epoch": [1, 3, 10],
    "batch_size": [1024],
    # "batch_size": [1024, 2048],
    # "batch_size": [256, 512, 1024],
    # "batch_size": [256, 512, 1024, 2048, 4096, 8192],
    "hidden_norm": [torch.nn.BatchNorm1d],
    # "hidden_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "norm_after_activation": [False],
    # "norm_after_activation": [False, True],
    "embedding_norm": [torch.nn.Identity],
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d],
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "initialization": [torch.nn.init.xavier_uniform_, torch.nn.init.xavier_normal_],
    "optim": [torch.optim.Adam],
    # "optim": [torch.optim.Adam, torch.optim.SGD],
}

FAST_PARAM_GRID = copy.deepcopy(PARAM_GRID)
FAST_PARAM_GRID.update(
    {
        "max_epochs": [10, 50],
        "check_val_every_n_epoch": [3, 10],
    }
)

FASTER_PARAM_GRID = copy.deepcopy(PARAM_GRID)
FASTER_PARAM_GRID.update(
    {
        "hidden_layers": [0, 1],
        "hidden_dim": [64, 128],
        "patience": [1, 3],
        "max_epochs": [10],
        "check_val_every_n_epoch": [1],
    }
)

# These are good for dcase, change for other event-based secret tasks
EVENT_POSTPROCESSING_GRID = {
    "median_filter_ms": [250],
    "min_duration": [125, 250],
    #    "median_filter_ms": [0, 62, 125, 250, 500, 1000],
    #    "min_duration": [0, 62, 125, 250, 500, 1000],
}

NUM_WORKERS = int(multiprocessing.cpu_count() / (max(1, torch.cuda.device_count())))


class OneHotToCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # One and only one label per class
        assert torch.all(
            torch.sum(y, dim=1) == torch.ones(y.shape[0], device=self.device)
        )
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


class FullyConnectedPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, prediction_type: str, conf: Dict):
        super().__init__()

        hidden_modules: List[torch.nn.Module] = []
        curdim = nfeatures
        # Honestly, we don't really know what activation preceded
        # us for the final embedding.
        last_activation = "linear"
        if conf["hidden_layers"]:
            for i in range(conf["hidden_layers"]):
                linear = torch.nn.Linear(curdim, conf["hidden_dim"])
                conf["initialization"](
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                hidden_modules.append(torch.nn.Dropout(conf["dropout"]))
                hidden_modules.append(torch.nn.ReLU())
                if conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                curdim = conf["hidden_dim"]
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore
        self.projection = torch.nn.Linear(curdim, nlabels)

        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )
        self.logit_loss: torch.nn.Module
        if prediction_type == "multilabel":
            self.activation: torch.nn.Module = torch.nn.Sigmoid()
            self.logit_loss = torch.nn.BCEWithLogitsLoss()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
            self.logit_loss = OneHotToCrossEntropyLoss()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_logit(x)
        x = self.activation(x)
        return x


class AbstractPredictionModel(pl.LightningModule):
    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[ScoreFunction],
        conf: Dict,
    ):
        super().__init__()

        self.save_hyperparameters(conf)

        # Since we don't know how these embeddings are scaled
        self.layernorm = conf["embedding_norm"](nfeatures)
        self.predictor = FullyConnectedPrediction(
            nfeatures, nlabels, prediction_type, conf
        )
        torchinfo.summary(self.predictor, input_size=(64, nfeatures))
        self.label_to_idx = label_to_idx
        self.idx_to_label: Dict[int, str] = {
            idx: label for (label, idx) in self.label_to_idx.items()
        }
        self.scores = scores

    def forward(self, x):
        # x = self.layernorm(x)
        x = self.predictor(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, _ = batch
        y_hat = self.predictor.forward_logit(x)
        loss = self.predictor.logit_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def _step(self, batch, batch_idx):
        # -> Dict[str, Union[torch.Tensor, List(str)]]:
        x, y, metadata = batch
        y_hat = self.predictor.forward_logit(x)
        y_pr = self.predictor(x)
        z = {
            "prediction": y_pr,
            "prediction_logit": y_hat,
            "target": y,
        }
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        return {**z, **metadata}

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    # Implement this for each inheriting class
    # TODO: Can we combine the boilerplate for both of these?
    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        """
        Return at the end of every validation and test epoch.
        :param name: "val" or "test"
        :param outputs: Unflattened minibatches from {name}_step,
            each with "target", "prediction", and additional metadata,
            with a list of values for each instance in the batch.
        :return:
        """
        raise NotImplementedError("Implement this in children")

    def validation_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("test", outputs)

    def _flatten_batched_outputs(
        self,
        outputs,  #: Union[torch.Tensor, List[str]],
        keys: List[str],
        dont_stack: List[str] = [],
    ) -> Dict:
        # ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        flat_outputs_default: DefaultDict = defaultdict(list)
        for output in outputs:
            assert set(output.keys()) == set(keys), f"{output.keys()} != {keys}"
            for key in keys:
                flat_outputs_default[key] += output[key]
        flat_outputs = dict(flat_outputs_default)
        for key in keys:
            if key in dont_stack:
                continue
            else:
                flat_outputs[key] = torch.stack(flat_outputs[key])
        return flat_outputs

    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr)
        return optimizer


class ScenePredictionModel(AbstractPredictionModel):
    """
    Prediction model with simple scoring over entire audio scenes.
    """

    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[ScoreFunction],
        conf: Dict,
    ):
        super().__init__(
            nfeatures=nfeatures,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=prediction_type,
            scores=scores,
            conf=conf,
        )

    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        flat_outputs = self._flatten_batched_outputs(
            outputs, keys=["target", "prediction", "prediction_logit"]
        )
        target, prediction, prediction_logit = (
            flat_outputs[key] for key in ["target", "prediction", "prediction_logit"]
        )

        end_scores = {}
        end_scores[f"{name}_loss"] = self.predictor.logit_loss(prediction_logit, target)

        if name == "test":
            # Cache all predictions for later serialization
            self.test_predicted_labels = prediction

        for score in self.scores:
            end_scores[f"{name}_{score}"] = score(
                prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
            )
        self.log(
            f"{name}_score", end_scores[f"{name}_{str(self.scores[0])}"], logger=True
        )
        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True, logger=True)


class EventPredictionModel(AbstractPredictionModel):
    """
    Event prediction model. For validation (and test),
    we combine timestamp events that are adjacent,
    but discard ones that are too short.
    """

    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[ScoreFunction],
        validation_target_events: Dict[str, List[Dict[str, Any]]],
        test_target_events: Dict[str, List[Dict[str, Any]]],
        conf: Dict,
    ):
        super().__init__(
            nfeatures=nfeatures,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=prediction_type,
            scores=scores,
            conf=conf,
        )
        self.target_events = {
            "val": validation_target_events,
            "test": test_target_events,
        }
        # For each epoch, what postprocessing parameters were best
        self.epoch_best_postprocessing: Dict[int, Tuple[Tuple[str, Any], ...]] = {}

    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        flat_outputs = self._flatten_batched_outputs(
            outputs,
            keys=["target", "prediction", "prediction_logit", "filename", "timestamp"],
            # This is a list of string, not tensor, so we don't need to stack it
            dont_stack=["filename"],
        )
        target, prediction, prediction_logit, filename, timestamp = (
            flat_outputs[key]
            for key in [
                "target",
                "prediction",
                "prediction_logit",
                "filename",
                "timestamp",
            ]
        )

        if name == "val":
            # During training, the epoch is one behind the value that will
            # be stored as the "best" epoch
            epoch = self.current_epoch + 1
            postprocessing_cached = None
        elif name == "test":
            epoch = self.current_epoch
            postprocessing_cached = self.epoch_best_postprocessing[epoch]
        else:
            raise ValueError
        # print("\n\n\n", epoch)

        predicted_events_by_postprocessing = get_events_for_all_files(
            prediction, filename, timestamp, self.idx_to_label, postprocessing_cached
        )

        score_and_postprocessing = []
        for postprocessing in tqdm(predicted_events_by_postprocessing):
            predicted_events = predicted_events_by_postprocessing[postprocessing]
            primary_score_fn = self.scores[0]
            primary_score = primary_score_fn(
                # predicted_events, self.target_events[name]
                predicted_events,
                self.target_events[name],
            )
            if np.isnan(primary_score):
                primary_score = 0.0
            score_and_postprocessing.append((primary_score, postprocessing))
        score_and_postprocessing.sort(reverse=True)

        # for vs in score_and_postprocessing:
        #    print(vs)

        best_postprocessing = score_and_postprocessing[0][1]
        if name == "val":
            print("BEST POSTPROCESSING", best_postprocessing)
            for k, v in best_postprocessing:
                self.log(f"postprocessing/{k}", v, logger=True)
            self.epoch_best_postprocessing[epoch] = best_postprocessing
        predicted_events = predicted_events_by_postprocessing[best_postprocessing]

        end_scores = {}
        end_scores[f"{name}_loss"] = self.predictor.logit_loss(prediction_logit, target)

        if name == "test":
            # print("test epoch", self.current_epoch)
            # Cache all predictions for later serialization
            self.test_predicted_labels = prediction
            self.test_predicted_events = predicted_events

        for score in self.scores:
            end_scores[f"{name}_{score}"] = score(
                predicted_events, self.target_events[name]
            )
            # Weird, this can happen if precision has zero guesses
            if math.isnan(end_scores[f"{name}_{score}"]):
                end_scores[f"{name}_{score}"] = 0.0
        self.log(
            f"{name}_score", end_scores[f"{name}_{str(self.scores[0])}"], logger=True
        )

        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True, logger=True)


class SplitMemmapDataset(Dataset):
    """
    Embeddings are memmap'ed, unless in-memory = True.

    WARNING: Don't shuffle this or access will be SLOW.
    """

    def __init__(
        self,
        embedding_path: Path,
        label_to_idx: Dict[str, int],
        nlabels: int,
        split_name: str,
        embedding_type: str,
        in_memory: bool,
        metadata: bool,
    ):
        self.embedding_path = embedding_path
        self.label_to_idx = label_to_idx
        self.nlabels = nlabels
        self.split_name = split_name
        self.embedding_type = embedding_type

        self.dim = tuple(
            json.load(
                open(embedding_path.joinpath(f"{split_name}.embedding-dimensions.json"))
            )
        )
        self.embeddings = np.memmap(
            filename=embedding_path.joinpath(f"{split_name}.embeddings.npy"),
            dtype=np.float32,
            mode="r",
            shape=self.dim,
        )
        if in_memory:
            self.embeddings = torch.stack(
                [torch.tensor(e) for e in tqdm(self.embeddings)]
            )
        self.labels = pickle.load(
            open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        )
        # Only used for event-based prediction, for validation and test scoring,
        # For timestamp (event) embedding tasks,
        # the metadata for each instance is {filename: , timestamp: }.
        if self.embedding_type == "event" and metadata:
            filename_timestamps_json = embedding_path.joinpath(
                f"{split_name}.filename-timestamps.json"
            )
            self.metadata = [
                {"filename": filename, "timestamp": timestamp}
                for filename, timestamp in json.load(open(filename_timestamps_json))
            ]
        else:
            self.metadata = [{}] * self.dim[0]
        assert len(self.labels) == self.dim[0]
        assert len(self.labels) == len(self.embeddings)
        assert len(self.labels) == len(self.metadata)
        assert self.embeddings[0].shape[0] == self.dim[1]

        """
        For all labels, return a multi or one-hot vector.
        This allows us to have tensors that are all the same shape.
        Later we reduce this with an argmax to get the vocabulary indices.
        """
        ys = []
        for idx in tqdm(range(len(self.labels))):
            labels = [self.label_to_idx[str(label)] for label in self.labels[idx]]
            y = label_to_binary_vector(labels, self.nlabels)
            ys.append(y)
        self.y = torch.stack(ys)
        assert self.y.shape == (len(self.labels), self.nlabels)

    def __len__(self) -> int:
        return self.dim[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.embeddings[idx], self.y[idx], self.metadata[idx]


def create_events_from_prediction(
    prediction_dict: Dict[float, torch.Tensor],
    idx_to_label: Dict[int, str],
    threshold: float = 0.5,
    median_filter_ms: float = 150,
    min_duration: float = 60.0,
) -> List[Dict[str, Union[float, str]]]:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events.
    (This is for one particular audio scene.)
    We convert the prediction tensor to a binary label based on the threshold value. Any
    events occurring at adjacent timestamps are considered to be part of the same event.
    This loops through and creates events for each label class.
    We optionally apply median filtering to predictions.
    We disregard events that are less than the min_duration milliseconds.

    Args:
        prediction_dict: A dictionary of predictions keyed on timestamp
            {timestamp -> prediction}. The prediction is a tensor of label
            probabilities.
        idx_to_label: Index to label mapping.
        threshold: Threshold for determining whether to apply a label
        min_duration: the minimum duration in milliseconds for an
                event to be included.

    Returns:
        A list of dicts withs keys "label", "start", and "end"
    """
    # Make sure the timestamps are in the correct order
    timestamps = np.array(sorted(prediction_dict.keys()))

    # Create a sorted numpy matrix of frame level predictions for this file. We convert
    # to a numpy array here before applying a median filter.
    predictions = np.stack(
        [prediction_dict[t].detach().cpu().numpy() for t in timestamps]
    )

    # Optionally apply a median filter here to smooth out events.
    ts_diff = np.mean(np.diff(timestamps))
    if median_filter_ms:
        filter_width = int(round(median_filter_ms / ts_diff))
        if filter_width:
            predictions = median_filter(predictions, size=(filter_width, 1))

    # Convert probabilities to binary vectors based on threshold
    predictions = (predictions > threshold).astype(np.int8)

    events = []
    for label in range(predictions.shape[1]):
        for group in more_itertools.consecutive_groups(
            np.where(predictions[:, label])[0]
        ):
            grouptuple = tuple(group)
            assert (
                tuple(sorted(grouptuple)) == grouptuple
            ), f"{sorted(grouptuple)} != {grouptuple}"
            startidx, endidx = (grouptuple[0], grouptuple[-1])

            start = timestamps[startidx]
            end = timestamps[endidx]
            # Add event if greater than the minimum duration threshold
            if end - start >= min_duration:
                events.append(
                    {"label": idx_to_label[label], "start": start, "end": end}
                )

    # This is just for pretty output, not really necessary
    events.sort(key=lambda k: k["start"])
    return events


def get_events_for_all_files(
    predictions: torch.Tensor,
    filenames: List[str],
    timestamps: torch.Tensor,
    idx_to_label: Dict[int, str],
    postprocessing: Optional[Tuple[Tuple[str, Any], ...]] = None,
) -> Dict[Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[str, float]]]]]:
    """
    Produces lists of events from a set of frame based label probabilities.
    The input prediction tensor may contain frame predictions from a set of different
    files concatenated together. file_timestamps has a list of filenames and
    timestamps for each frame in the predictions tensor.

    We split the predictions into separate tensors based on the filename and compute
    events based on those individually.

    If no postprocessing is specified (during training), we try a
    variety of ways of postprocessing the predictions into events,
    including median filtering and minimum event length.

    If postprocessing is specified (during test, chosen at the best
    validation epoch), we use this postprocessing.

    Args:
        predictions: a tensor of frame based multi-label predictions.
        filenames: a list of filenames where each entry corresponds
            to a frame in the predictions tensor.
        timestamps: a list of timestamps where each entry corresponds
            to a frame in the predictions tensor.
        idx_to_label: Index to label mapping.
        postprocessing: See above.

    Returns:
        A dictionary from filtering params to the following values:
        A dictionary of lists of events keyed on the filename slug.
        The event list is of dicts of the following format:
            {"label": str, "start": float ms, "end": float ms}
    """
    # This probably could be more efficient if we make the assumption that
    # timestamps are in sorted order. But this makes sure of it.
    assert predictions.shape[0] == len(filenames)
    assert predictions.shape[0] == len(timestamps)
    event_files: Dict[str, Dict[float, torch.Tensor]] = {}
    for i, (filename, timestamp) in enumerate(zip(filenames, timestamps)):
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in event_files:
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][float(timestamp)] = predictions[i]

    # Create events for all the different files. Store all the events as a dictionary
    # with the same format as the ground truth from the luigi pipeline.
    # Ex) { slug -> [{"label" : "woof", "start": 0.0, "end": 2.32}, ...], ...}
    event_dict: Dict[
        Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[float, str]]]]
    ] = {}
    if postprocessing:
        postprocess = postprocessing
        event_dict[postprocess] = {}
        for slug, timestamp_predictions in event_files.items():
            event_dict[postprocess][slug] = create_events_from_prediction(
                timestamp_predictions, idx_to_label, **dict(postprocess)
            )
    else:
        postprocessing_confs = list(ParameterGrid(EVENT_POSTPROCESSING_GRID))
        for postprocess_dict in tqdm(postprocessing_confs):
            postprocess = tuple(postprocess_dict.items())
            event_dict[postprocess] = {}
            for slug, timestamp_predictions in event_files.items():
                event_dict[postprocess][slug] = create_events_from_prediction(
                    timestamp_predictions, idx_to_label, **postprocess_dict
                )

    return event_dict


def label_vocab_nlabels(embedding_path: Path) -> Tuple[pd.DataFrame, int]:
    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    return (label_vocab, nlabels)


def dataloader_from_split_name(
    split_name: str,
    embedding_path: Path,
    label_to_idx: Dict[str, int],
    nlabels: int,
    embedding_type: str,
    in_memory: bool,
    metadata: bool = True,
    batch_size: int = 64,
) -> DataLoader:
    dataset = SplitMemmapDataset(
        embedding_path=embedding_path,
        label_to_idx=label_to_idx,
        nlabels=nlabels,
        split_name=split_name,
        embedding_type=embedding_type,
        in_memory=in_memory,
        metadata=metadata,
    )

    print(
        f"Getting embeddings for split {split_name}, "
        + f"which has {len(dataset)} instances."
    )

    if in_memory:
        num_workers = NUM_WORKERS
    else:
        # We are disk bound, so multiple workers might cause thrashing
        num_workers = 0

    if in_memory and split_name == "train":
        shuffle = True
    else:
        # We don't shuffle if we are memmap'ing from disk
        # We don't shuffle validation and test, to maintain the order
        # of the event metadata
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )


class GetPartitionDataLoaders:
    def __init__(
        self,
        metadata: Dict[str, Any],
        embedding_path: Path,
        label_to_idx: Dict[str, int],
        nlabels: int,
        embedding_type: str,
        in_memory: bool,
        metadata_req: bool = True,
    ):
        self.metadata = metadata
        self.embedding_path = embedding_path
        self.label_to_idx = label_to_idx
        self.nlabels = nlabels
        self.embedding_type = embedding_type
        self.in_memory = in_memory
        self.metadata_req = metadata_req
        if in_memory:
            self.num_workers = NUM_WORKERS
        else:
            # We are disk bound, so multiple workers might cause thrashing
            self.num_workers = 0

        # Save already loaded datasets in this variable to avoid reloading datasets
        # for constructing different sets
        self.loaded_datasets = {}

    def get_split_dataloader(
        self, split: str, fold_names: List[str], batch_size: int
    ) -> DataLoader:
        """
        Gets the dataloader of the split by loading data for each fold in
        the split.
        """

        datasets: List[Dataset] = []
        for fold_name in fold_names:
            # If the dataset has already been loaded, rather than loading it again,
            # fetch it from the loaded_datasets
            if fold_name in self.loaded_datasets:
                dataset = self.loaded_datasets[fold_name]
            else:
                dataset = SplitMemmapDataset(
                    embedding_path=self.embedding_path,
                    label_to_idx=self.label_to_idx,
                    nlabels=self.nlabels,
                    split_name=fold_name,
                    embedding_type=self.embedding_type,
                    in_memory=self.in_memory,
                    metadata_req=self.metadata_req,
                )
                # Save a reference to the loaded dataset so that the dataset can be
                # reused in another partition. This also helps to avoid loading
                # the dataset again while geting dataloaders with different batch size
                # during the grid search
                self.loaded_datasets[fold_name] = dataset
            datasets.append(dataset)

        # Combine the datasets for the split and create the dataloader from the
        # combined dataset
        combined_dataset = ConcatDataset(datasets) if len(datasets) > 0 else datasets[0]
        if self.in_memory and split in ["train", "train+dev"]:
            shuffle = True
        else:
            # We don't shuffle if we are memmap'ing from disk
            # We don't shuffle validation and test, to maintain the order
            # of the event metadata
            shuffle = False

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def get_partition_dataloaders(
        self, batch_size: int = 64
    ) -> List[Dict[str, DataLoader]]:
        """
        Generate the split dataloaders for each partition of the dataset.

        If the data has multiple folds, each partition is Leave One Out Cross
        Validation set i.e. one fold is used in the test split and all other
        folds are used in the train+dev split.

        If the data has no folds, and the test, train and valid sets are already
        defined, a single partition with the name `no_folds` will be returned.
        """
        all_partition_folds: Dict[str, Dict[str, List[str]]] = {}
        all_partition_dataloaders: List[str, Dict[str, DataLoader]] = {}
        if self.metadata["mode"] == "folds":
            num_folds = self.metadata["num_folds"]  # 4
            folds = self.metadata["folds"]  # ["fold0", "fold1", "fold2", "fold3"]
            for fold_idx in range(num_folds):
                partition_name = f"fold_{fold_idx}"
                test_fold = folds[fold_idx]
                dev_fold = folds[(fold_idx + 1) % num_folds]
                train_folds = set(folds) - {test_fold, dev_fold}
                all_partition_folds[partition_name] = {
                    "train": train_folds,
                    "dev": [dev_fold],
                    "train+dev": train_folds + [dev_fold],
                    "test": [test_fold],
                }
                assert not train_folds.intersection(
                    {test_fold, dev_fold}
                ), "Train folds are not distinct from the dev and the test folds"
        else:
            # If the mode of dataset is not folds, make only one partition and
            # the name of the partition is no_folds
            all_partition_folds["single_fold"] = {
                "train": ["train"],
                "dev": ["valid"],
                "train+dev": ["train", "valid"],
                "test": ["test"],
            }

        all_partition_dataloaders = {
            partition_name: {
                split: self.get_split_dataloader(split, fold_names, batch_size)
                for split, fold_names in partition_folds.items()
            }
            for partition_name, partition_folds in all_partition_folds.items()
        }

        return all_partition_dataloaders


class GridPointResult:
    def __init__(
        self,
        model_path: str,
        epoch: int,
        time_in_min: float,
        hparams: Dict[str, Any],
        postprocessing: Tuple[Tuple[str, Any], ...],
        trainer: pl.Trainer,
        validation_score: float,
        score_mode: str,
    ):
        self.model_path = model_path
        self.epoch = epoch
        self.time_in_min = time_in_min
        self.hparams = hparams
        self.postprocessing = postprocessing
        self.trainer = trainer
        self.validation_score = validation_score
        self.score_mode = score_mode


def task_predictions_train(
    embedding_path: Path,
    embedding_size: int,
    grid_points: int,
    metadata: Dict[str, Any],
    label_to_idx: Dict[str, int],
    nlabels: int,
    scores: List[ScoreFunction],
    conf: Dict,
    gpus: Any,
    in_memory: bool,
    deterministic: bool,
) -> GridPointResult:
    start = time.time()
    predictor: AbstractPredictionModel
    if metadata["embedding_type"] == "event":
        validation_target_events = json.load(
            embedding_path.joinpath("valid.json").open()
        )
        test_target_events = json.load(embedding_path.joinpath("test.json").open())
        predictor = EventPredictionModel(
            nfeatures=embedding_size,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=metadata["prediction_type"],
            scores=scores,
            validation_target_events=validation_target_events,
            test_target_events=test_target_events,
            conf=conf,
        )
    elif metadata["embedding_type"] == "scene":
        predictor = ScenePredictionModel(
            nfeatures=embedding_size,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=metadata["prediction_type"],
            scores=scores,
            conf=conf,
        )
    else:
        raise ValueError(f"Unknown embedding_type {metadata['embedding_type']}")

    # First score is the target
    target_score = f"val_{str(scores[0])}"

    if scores[0].maximize:
        mode = "max"
    else:
        mode = "min"
    checkpoint_callback = ModelCheckpoint(monitor=target_score, mode=mode)
    early_stop_callback = EarlyStopping(
        monitor=target_score,
        min_delta=0.00,
        patience=conf["patience"],
        check_on_train_epoch_end=False,
        verbose=False,
        mode=mode,
    )

    logger = CSVLogger(Path("logs").joinpath(embedding_path))
    logger.log_hyperparams(hparams_to_json(conf))

    # Try also pytorch profiler
    # profiler = pl.profiler.AdvancedProfiler(output_filename="predictions-profile.txt")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        gpus=gpus,
        check_val_every_n_epoch=conf["check_val_every_n_epoch"],
        max_epochs=conf["max_epochs"],
        deterministic=deterministic,
        num_sanity_val_steps=0,
        # profiler=profiler,
        # profiler="pytorch",
        profiler="simple",
        logger=logger,
    )
    train_dataloader = dataloader_from_split_name(
        "train",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
        in_memory=in_memory,
        metadata=False,
    )
    valid_dataloader = dataloader_from_split_name(
        "valid",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
        in_memory=in_memory,
    )
    trainer.fit(predictor, train_dataloader, valid_dataloader)
    if checkpoint_callback.best_model_score is not None:
        sys.stdout.flush()
        end = time.time()
        time_in_min = (end - start) / 60
        epoch = torch.load(checkpoint_callback.best_model_path)["epoch"]
        if metadata["embedding_type"] == "event":
            best_postprocessing = predictor.epoch_best_postprocessing[epoch]
        else:
            best_postprocessing = []
        # TODO: Postprocessing
        logger.log_metrics({"time_in_min": time_in_min})
        logger.finalize("success")
        logger.save()
        return GridPointResult(
            model_path=checkpoint_callback.best_model_path,
            epoch=epoch,
            time_in_min=time_in_min,
            hparams=dict(predictor.hparams),
            postprocessing=best_postprocessing,
            trainer=trainer,
            validation_score=checkpoint_callback.best_model_score.detach().cpu().item(),
            score_mode=mode,
        )
    else:
        raise ValueError(
            f"No score {checkpoint_callback.best_model_score} for this model"
        )


def serialize_value(v):
    if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
        return v
    else:
        return str(v)


def hparams_to_json(hparams):
    return {k: serialize_value(v) for k, v in hparams.items()}


def task_predictions(
    embedding_path: Path,
    embedding_size: int,
    grid_points: int,
    gpus: Optional[int],
    in_memory: bool,
    deterministic: bool,
    grid: str,
):
    # By setting workers=True in seed_everything(), Lightning derives
    # unique seeds across all dataloader workers and processes
    # for torch, numpy and stdlib random number generators.
    # Note that if you change the number of workers, determinism
    # might change.
    # However, it appears that workers=False does get deterministic
    # results on 4 multi-worker jobs I ran, probably because our
    # dataloader doesn't do any augmentation or use randomness.
    if deterministic:
        seed_everything(42, workers=False)

    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)

    # wandb.init(project="heareval", tags=["predictions", embedding_path.name])

    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")
    scores = [
        available_scores[score](label_to_idx=label_to_idx)
        for score in metadata["evaluation"]
    ]

    def sort_grid_points(grid_point_results: List[GridPointResult]) -> None:
        """
        Sort grid point results in place, so that the first result
        is the best.
        """
        # TODO: Assert all score modes are the same?
        mode = grid_point_results[0].score_mode
        # Pick the model with the best validation score
        grid_point_results.sort(key=lambda g: -g.validation_score)
        if mode == "max":
            pass
        elif mode == "min":
            grid_point_results.reverse()
        else:
            raise ValueError(f"mode = {mode}")

    def print_scores(grid_point_results: List[GridPointResult]):
        sort_grid_points(grid_point_results)
        for g in grid_point_results:
            print(
                json.dumps(
                    (
                        g.validation_score,
                        g.epoch,
                        hparams_to_json(g.hparams),
                        g.postprocessing,
                        str(embedding_path),
                    )
                )
            )

    if grid == "default":
        final_grid = copy.copy(PARAM_GRID)
    elif grid == "fast":
        final_grid = copy.copy(FAST_PARAM_GRID)
    elif grid == "faster":
        final_grid = copy.copy(FASTER_PARAM_GRID)
    if metadata["task_name"] in TASK_SPECIFIC_PARAM_GRID:
        final_grid.update(TASK_SPECIFIC_PARAM_GRID[metadata["task_name"]])

    grid_point_results = []
    # Model selection
    confs = list(ParameterGrid(final_grid))
    random.shuffle(confs)
    for conf in tqdm(confs[:grid_points], desc="grid"):
        print("trying grid point", conf)
        grid_point_result = task_predictions_train(
            embedding_path=embedding_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            metadata=metadata,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            scores=scores,
            conf=conf,
            gpus=gpus,
            in_memory=in_memory,
            deterministic=deterministic,
        )
        grid_point_results.append(grid_point_result)
        print_scores(grid_point_results)

    # Use the best model to compute test scores
    sort_grid_points(grid_point_results)
    best_grid_point = grid_point_results[0]
    print()
    print(
        "Best validation score",
        best_grid_point.validation_score,
        best_grid_point.hparams,
        embedding_path,
    )
    print(best_grid_point.model_path)

    test_dataloader = dataloader_from_split_name(
        "test",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
        in_memory=in_memory,
    )
    best_trainer = best_grid_point.trainer
    # This hack is necessary because we use the best validation epoch to
    # choose the event postprocessing
    best_trainer.fit_loop.current_epoch = best_grid_point.epoch

    test_results = best_trainer.test(
        ckpt_path=best_grid_point.model_path, test_dataloaders=test_dataloader
    )
    assert len(test_results) == 1, "Should have only one test dataloader"
    test_results = test_results[0]

    test_results.update(
        {
            "validation_score": best_grid_point.validation_score,
            "hparams": hparams_to_json(best_grid_point.hparams),
            "postprocessing": best_grid_point.postprocessing,
            "epoch": best_grid_point.epoch,
            "time_in_min": best_grid_point.time_in_min,
            "score_mode": best_grid_point.score_mode,
            "embedding_path": str(embedding_path),
        }
    )
    open(embedding_path.joinpath("test.predicted-scores.json"), "wt").write(
        json.dumps(test_results, indent=4)
    )
    print("TEST RESULTS", json.dumps(test_results))

    # We no longer have best_predictor, the predictor is
    # loaded by trainer.test and then disappears
    """
    # Cache predictions for secondary sanity-check evaluation
    if metadata["embedding_type"] == "event":
        json.dump(
            best_predictor.test_predicted_events,
            embedding_path.joinpath("test.predictions.json").open("w"),
            indent=4,
        )
    pickle.dump(
        best_predictor.test_predicted_labels,
        open(embedding_path.joinpath("test.predicted-labels.pkl"), "wb"),
    )
    """
