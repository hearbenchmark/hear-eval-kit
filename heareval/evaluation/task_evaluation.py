#!/usr/bin/env python3
"""
Compute evaluation metrics on a set of predictions for a task.

TODO: This is a start on this file. Still need to add a bunch of metrics for the tasks
    including AUC, chroma error for NSYnth pitch, and event-based metrics for DCASE.
    Would it make sense to move all the metric functions over to a file called metrics?
"""

from functools import partial
import json
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

from dcase_util.containers import MetaDataContainer
import numpy as np
import pandas as pd
import sed_eval
import sklearn.metrics
import torch


def top1_error(
    predictions: np.ndarray, targets: List, label_vocab: pd.DataFrame
) -> Dict[str, float]:

    # Dictionary of labels and integer idx: {label -> idx}
    label_vocab = label_vocab.set_index("label").to_dict()["idx"]

    # Compute the number of correct predictions
    correct = 0
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        assert len(targets[i]) == 1
        target_class = label_vocab[targets[i][0]]

        if predicted_class == target_class:
            correct += 1

    top1_error = correct / len(targets)
    return {"top1_error": top1_error}


def macroauc(
    predictions: np.ndarray, targets: List, label_vocab: pd.DataFrame
) -> Dict[str, float]:
    #    return {"macroauc": 0.0}
    # The rest is broken if test set vocabulary is a strict subset of train vocabulary.
    # TODO: This should happen in task_evaluation, not be reused everywhere
    # Dictionary of labels and integer idx: {label -> idx}
    label_vocab = label_vocab.set_index("label").to_dict()["idx"]
    # TODO: This shape only works for multiclass, check that is the type
    for rowlabels in targets:
        assert len(rowlabels) == 1
    y_true = np.array([label_vocab[rowlabels[0]] for rowlabels in targets])

    # Convert to multilabel one-hot
    y_true_multilabel = np.zeros(predictions.shape)
    y_true_multilabel[np.arange(y_true.size), y_true] = 1

    import IPython

    ipshell = IPython.embed
    ipshell(banner1="ipshell")
    return {"macroauc": sklearn.metrics.roc_auc_score(y_true, predictions)}


def sed_eval_event_container(x: Dict) -> MetaDataContainer:
    # Reformat event list for sed_eval
    reference_events = []
    for filename, event_list in x.items():
        for event in event_list:
            reference_events.append(
                {
                    # Convert from ms to seconds for sed_eval
                    "event_label": event["label"],
                    "event_onset": event["start"] / 1000.0,
                    "event_offset": event["end"] / 1000.0,
                    "file": filename,
                }
            )
    return MetaDataContainer(reference_events)


def event_based_metrics(
    predictions,
    targets: Dict,
    label_vocab=pd.DataFrame,
    tolerance=0.200,
    onset=True,
    offset=True,
):
    """
    event-based metrics - the ground truth and system output are compared at
    event instance level;
    https://tut-arg.github.io/sed_eval/sound_event.html#sound-event-detection
    """
    # Containers of events for sed_eval
    reference_event_list = sed_eval_event_container(targets)
    estimated_event_list = sed_eval_event_container(predictions)

    metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=list(label_vocab["label"]),
        t_collar=tolerance,
        evaluate_onset=onset,
        evaluate_offset=offset,
    )

    for filename in predictions:
        metrics.evaluate(
            reference_event_list=reference_event_list.filter(filename=filename),
            estimated_event_list=estimated_event_list.filter(filename=filename),
        )

    # This (and segment_based_metrics) return a pretty large selection of metrics. We
    # might want to add a task_metadata option to filter these for the specific
    # metric that we are going to use to evaluate the task.
    overall_metrics = metrics.results_overall_metrics()
    return overall_metrics


def segment_based_metrics(
    predictions, targets: Dict, label_vocab=pd.DataFrame, time_resolution=1.0
):
    """
    segment-based metrics - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;
    https://tut-arg.github.io/sed_eval/sound_event.html#sound-event-detection
    """
    # Containers of events for sed_eval
    reference_event_list = sed_eval_event_container(targets)
    estimated_event_list = sed_eval_event_container(predictions)

    metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=list(label_vocab["label"]), time_resolution=time_resolution
    )

    for filename in predictions:
        metrics.evaluate(
            reference_event_list=reference_event_list.filter(filename=filename),
            estimated_event_list=estimated_event_list.filter(filename=filename),
        )

    overall_metrics = metrics.results_overall_metrics()
    return overall_metrics


# TODO: Add additional metrics

available_metrics = {
    "top1_error": top1_error,
    "macroauc": macroauc,
    "event_based": event_based_metrics,
    "onset_only_event_based": partial(event_based_metrics, offset=False),
    "segment_based": segment_based_metrics,
}


def get_scene_based_prediction_files(task_path: Path) -> Tuple[np.ndarray, List]:
    # Predictions are currently torch tensors -- should we convert to np arrays before
    # pickling?
    predictions = pickle.load(
        task_path.joinpath("test.predicted-labels.pkl").open("rb")
    )
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    elif isinstance(predictions, np.ndarray):
        pass
    else:
        raise TypeError(
            "Expected predictions to be a numpy array or a torch tensor. "
            f"Received: {type(predictions)}."
        )

    targets = pickle.load(task_path.joinpath("test.target-labels.pkl").open("rb"))

    # What other types could we receive as targets?
    assert isinstance(targets, list)

    # Make sure we have the same number of predictions as targets
    assert len(predictions) == len(targets)

    return predictions, targets


def get_event_based_prediction_files(task_path: Path) -> Tuple[Dict, Dict]:
    # For event based embeddings we load JSON files of the labeled sound events
    predictions = json.load(task_path.joinpath("test.predictions.json").open())
    targets = json.load(task_path.joinpath("test.json").open())
    assert isinstance(predictions, dict)
    assert isinstance(targets, dict)
    for filename in predictions:
        assert filename in targets

    return predictions, targets


def task_evaluation(task_path: Path):

    metadata = json.load(task_path.joinpath("task_metadata.json").open())
    label_vocab = pd.read_csv(task_path.joinpath("labelvocabulary.csv"))

    embedding_type = metadata["embedding_type"]

    if "evaluation" not in metadata:
        print(f"Task {task_path.name} has no evaluation config.")
        return

    if embedding_type == "scene":
        predictions, targets = get_scene_based_prediction_files(task_path)
    elif embedding_type == "event":
        predictions, targets = get_event_based_prediction_files(task_path)
    else:
        raise ValueError(f"Unknown embedding type received: {embedding_type}")

    metrics = metadata["evaluation"]
    results = {}
    for metric in metrics:
        print("  -", metric)
        new_results = available_metrics[metric](predictions, targets, label_vocab)
        results.update({metric: new_results})

    return results
