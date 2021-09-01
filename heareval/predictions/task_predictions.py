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

import json
import math
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

PARAM_GRID = {
    "hidden_layers": [0, 1, 2, 3],
    # "hidden_dim": [256, 512, 1024],
    "hidden_dim": [1024, 512],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    "patience": [20],
    "max_epochs": [500],
    "check_val_every_n_epoch": [1, 3, 10],
    "batch_size": [1024, 2048, 4096, 8192],
    "hidden_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "norm_after_activation": [False, True],
    "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "initialization": [torch.nn.init.xavier_uniform_, torch.nn.init.xavier_normal_],
    "optim": [torch.optim.Adam],
    # "optim": [torch.optim.Adam, torch.optim.SGD],
}

# These are good for dcase, change for other event-based secret tasks
EVENT_POSTPROCESSING_GRID = {
    "median_filter_ms": [0.0, 60.0, 150.0, 250.0],
    "min_duration": [0.0, 60.0, 150.0, 250.0],
}


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
        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True)


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
            postprocessing = None
        elif name == "test":
            epoch = self.current_epoch
            postprocessing = self.epoch_best_postprocessing[epoch]
        else:
            raise ValueError
        # print("\n\n\n", epoch)

        predicted_events_by_postprocessing = get_events_for_all_files(
            prediction, filename, timestamp, self.idx_to_label, postprocessing
        )

        score_and_postprocessing = []
        for postprocessing in predicted_events_by_postprocessing:
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

        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True)


class SplitMemmapDataset(Dataset):
    """
    Embeddings are memmap'ed.

    WARNING: Don't shuffle this or access will be SLOW.
    """

    def __init__(
        self,
        embedding_path: Path,
        label_to_idx: Dict[str, int],
        nlabels: int,
        split_name: str,
        embedding_type: str,
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
        self.embedding_memmap = np.memmap(
            filename=embedding_path.joinpath(f"{split_name}.embeddings.npy"),
            dtype=np.float32,
            mode="r",
            shape=self.dim,
        )
        self.labels = pickle.load(
            open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        )
        # Only used for event-based prediction
        # For timestamp (event) embedding tasks,
        # the metadata for each instance is {filename: , timestamp: }.
        if self.embedding_type == "event":
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
        assert len(self.labels) == len(self.embedding_memmap)
        assert len(self.labels) == len(self.metadata)
        assert self.embedding_memmap[0].shape[0] == self.dim[1]

    def __len__(self) -> int:
        return self.dim[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        For all labels, return a multi or one-hot vector.
        This allows us to have tensors that are all the same shape.
        Later we reduce this with an argmax to get the vocabulary indices.
            We also include the filename and timestamp, which we need
        for evaluation of timestamp (event) tasks.
        We also return the metadata as a Dict.
        """
        x = self.embedding_memmap[idx]
        labels = [self.label_to_idx[str(label)] for label in self.labels[idx]]
        y = label_to_binary_vector(labels, self.nlabels)
        return x, y, self.metadata[idx]


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
        for slug, timestamp_predictions in tqdm(event_files.items()):
            event_dict[postprocess][slug] = create_events_from_prediction(
                timestamp_predictions, idx_to_label, **dict(postprocess)
            )
    else:
        postprocessing_confs = list(ParameterGrid(EVENT_POSTPROCESSING_GRID))
        for postprocess_dict in postprocessing_confs:
            postprocess = tuple(postprocess_dict.items())
            event_dict[postprocess] = {}
            for slug, timestamp_predictions in tqdm(event_files.items()):
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
    batch_size: int = 64,
) -> DataLoader:
    dataset = SplitMemmapDataset(
        embedding_path=embedding_path,
        label_to_idx=label_to_idx,
        nlabels=nlabels,
        split_name=split_name,
        embedding_type=embedding_type,
    )

    print(
        f"Getting embeddings for split {split_name}, "
        + f"which has {len(dataset)} instances."
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # We don't shuffle because it's slow.
        # Also we want predicted labels in the same order as
        # target labels.
        shuffle=False,
    )


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
    deterministic: bool,
) -> Tuple[
    str, int, Dict[str, Any], Tuple[Tuple[str, Any], ...], pl.Trainer, float, str
]:
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

    # Try also pytorch profiler
    # profiler = pl.profiler.AdvancedProfiler(output_filename="predictions-profile.txt")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        gpus=gpus,
        check_val_every_n_epoch=conf["check_val_every_n_epoch"],
        max_epochs=conf["max_epochs"],
        deterministic=deterministic,
        # profiler=profiler,
        # profiler="pytorch",
        profiler="simple",
    )
    train_dataloader = dataloader_from_split_name(
        "train",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
    )
    valid_dataloader = dataloader_from_split_name(
        "valid",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
    )
    trainer.fit(predictor, train_dataloader, valid_dataloader)
    if checkpoint_callback.best_model_score is not None:
        sys.stdout.flush()
        end = time.time()
        epoch = torch.load(checkpoint_callback.best_model_path)["epoch"]
        if metadata["embedding_type"] == "event":
            best_postprocessing = predictor.epoch_best_postprocessing[epoch]
        else:
            best_postprocessing = []
        print(
            "\n\n\nGRID POINT",
            embedding_path,
            checkpoint_callback.best_model_score.detach().cpu().item(),
            json.dumps(best_postprocessing),
            "epoch %d" % epoch,
            "time_in_min %.2f" % ((end - start) / 60),
            json.dumps(hparams_to_json(predictor.hparams)),
            "\n\n\n",
        )
        sys.stdout.flush()
        return (
            checkpoint_callback.best_model_path,
            epoch,
            dict(predictor.hparams),
            best_postprocessing,
            trainer,
            checkpoint_callback.best_model_score.detach().cpu().item(),
            mode,
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
    scene_embedding_size: int,
    timestamp_embedding_size: int,
    grid_points: int,
    gpus: Optional[int],
    deterministic: bool,
):
    # By setting workers=True in seed_everything(), Lightning derives
    # unique seeds across all dataloader workers and processes for
    # torch, numpy and stdlib random number generators. When turned
    # on, it ensures that e.g. data augmentations are not repeated
    # across workers.
    # This means we should keep dataloader workers consistent.
    if deterministic:
        seed_everything(42, workers=False)

    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)

    # wandb.init(project="heareval", tags=["predictions", embedding_path.name])

    if metadata["embedding_type"] == "scene":
        embedding_size = scene_embedding_size
    elif metadata["embedding_type"] == "event":
        embedding_size = timestamp_embedding_size
    else:
        raise ValueError(f"Unknown embedding type {metadata['embedding_type']}")

    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")
    scores = [
        available_scores[score](label_to_idx=label_to_idx)
        for score in metadata["evaluation"]
    ]

    def print_scores(mode, scores_and_trainers):
        # Pick the model with the best validation score
        scores_and_trainers.sort(key=lambda st: -st[0])
        if mode == "max":
            pass
        elif mode == "min":
            scores_and_trainers.reverse()
        else:
            raise ValueError(f"mode = {mode}")
        # print(mode)
        for score, _, epoch, _, hparams, postprocessing in scores_and_trainers:
            print(score, epoch, hparams, postprocessing)

    mode = None
    scores_and_trainers = []
    # Model selection
    confs = list(ParameterGrid(PARAM_GRID))
    random.shuffle(confs)
    for conf in tqdm(confs[:grid_points], desc="grid"):
        # TODO: Assert mode doesn't change?
        (
            model_path,
            epoch,
            hparams,
            postprocessing,
            trainer,
            best_model_score,
            mode,
        ) = task_predictions_train(
            embedding_path=embedding_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            metadata=metadata,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            scores=scores,
            conf=conf,
            gpus=gpus,
            deterministic=deterministic,
        )
        scores_and_trainers.append(
            (best_model_score, model_path, epoch, trainer, hparams, postprocessing)
        )
        print_scores(mode, scores_and_trainers)

    with open(embedding_path.joinpath("valid.hparams.jsonl"), "wt") as w:
        for score, _, _, _, hparams, _ in scores_and_trainers:
            w.write(json.dumps([score, hparams_to_json(hparams)]) + "\n")
    with open(embedding_path.joinpath("valid.postprocessing.jsonl"), "wt") as w:
        for score, _, _, _, _, postprocessing in scores_and_trainers:
            w.write(json.dumps([score, postprocessing]) + "\n")

    # Use that model to compute test scores
    (
        best_score,
        best_model_path,
        best_model_epoch,
        best_trainer,
        best_hparams,
        best_postprocessing,
    ) = scores_and_trainers[0]
    print()
    print("Best validation score", best_score, best_hparams)
    print(best_model_path)

    open(embedding_path.joinpath("test.best-model-config.json"), "wt").write(
        json.dumps([hparams_to_json(best_hparams), best_postprocessing], indent=4)
    )

    test_dataloader = dataloader_from_split_name(
        "test",
        embedding_path,
        label_to_idx,
        nlabels,
        metadata["embedding_type"],
        batch_size=conf["batch_size"],
    )
    # This hack is necessary because we use the best validation epoch to
    # choose the event postprocessing
    best_trainer.fit_loop.current_epoch = best_model_epoch

    test_scores = best_trainer.test(
        ckpt_path=best_model_path, test_dataloaders=test_dataloader
    )
    assert len(test_scores) == 1, "Should have only one test dataloader"
    test_scores = test_scores[0]

    open(embedding_path.joinpath("test.predicted-scores.json"), "wt").write(
        json.dumps(test_scores, indent=4)
    )

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
