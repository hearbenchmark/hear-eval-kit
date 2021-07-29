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
from typing import Any, Callable, Collection, Dict, List, Tuple


from dcase_util.containers import MetaDataContainer
import numpy as np
import pandas as pd
import sed_eval
import sklearn.metrics
import torch


class MetricFunction:
    """
    A simple abstract base class for metric functions
    """

    def __init__(self, task_metadata: Dict, label_vocab: pd.DataFrame):
        self.task_metadata = task_metadata
        assert "idx" in label_vocab.columns, "label_vocab missing idx column"
        assert "label" in label_vocab, "label_vocab missing label column"
        self.label_vocab = label_vocab

    def label_vocab_as_dict(self, key: str) -> Dict:
        """
        Returns a dictionary of the label vocabulary mapping the label column to
        the idx column. key sets whether the label or idx is the key in the dict. The
        other column will the the value.
        """
        if key == "label":
            value = "idx"
        else:
            assert key == "idx", "key argument must be either 'label' or 'idx'"
            value = "label"
        return self.label_vocab.set_index(key).to_dict()[value]

    def __call__(self, predictions: Any, targets: Any, **kwargs) -> Dict:
        """
        Compute the metric based on the predictions and targets. Return a dictionary
        of the results.
        """
        raise NotImplementedError("Inheriting classes must implement this function")


class Top1Error(MetricFunction):
    def __call__(
        self, predictions: np.ndarray, targets: List, **kwargs
    ) -> Dict[str, float]:
        # Dictionary of labels and integer idx: {label -> idx}
        label_vocab = self.label_vocab_as_dict(key="label")

        # Compute the number of correct predictions
        correct = 0
        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction)
            assert len(targets[i]) == 1
            target_class = label_vocab[targets[i][0]]

            if predicted_class == target_class:
                correct += 1

        error = correct / len(targets)
        return {"top1_error": error}


class MacroAUC(MetricFunction):
    def __call__(
        self, predictions: np.ndarray, targets: List, **kwargs
    ) -> Dict[str, float]:
        # return {"auc": None}
        # Dictionary of labels and integer idx: {label -> idx}
        label_vocab = self.label_vocab_as_dict(key="label")

        # TODO: This shape only works for multiclass, check that is the type
        if self.task_metadata["prediction_type"] == "multiclass":
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


class ChromaError(MetricFunction):
    """
    Metric specifically for pitch detection -- converts all pitches to chroma first.
    This metric ignores octave errors in pitch classification.
    """

    def __call__(
        self, predictions: np.ndarray, targets: List, **kwargs
    ) -> Dict[str, float]:
        # Dictionary of labels and integer idx: {label -> idx}
        label_vocab = self.label_vocab_as_dict(key="label")

        # Compute the number of correct predictions
        correct = 0
        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction)
            assert len(targets[i]) == 1
            target_class = label_vocab[targets[i][0]]
            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        error = correct / len(targets)
        return {"chroma_error": error}


class SoundEventMetric(MetricFunction):
    """
    Metrics for sound event detection tasks using sed_eval
    """

    # Metric class must be defined in inheriting classes
    metric_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self, task_metadata: Dict, label_vocab: pd.DataFrame, params: Dict = None
    ):
        super().__init__(task_metadata=task_metadata, label_vocab=label_vocab)
        self.params = params if params is not None else {}
        assert self.metric_class is not None

    def __call__(self, predictions: Dict, targets: Dict, **kwargs):
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        metrics = self.metric_class(
            event_label_list=list(self.label_vocab["label"]), **self.params
        )

        for filename in predictions:
            metrics.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # This (and segment_based_metrics) return a pretty large selection of metrics.
        # We might want to add a task_metadata option to filter these for the specific
        # metric that we are going to use to evaluate the task.
        overall_metrics = metrics.results_overall_metrics()
        return overall_metrics

    @staticmethod
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


class SegmentBasedMetric(SoundEventMetric):
    """
    segment-based metrics - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    metric_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedMetric(SoundEventMetric):
    """
    event-based metrics - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    metric_class = sed_eval.sound_event.EventBasedMetrics


available_metrics: Dict[str, Callable] = {
    "top1_error": Top1Error,
    "macroauc": MacroAUC,
    "chroma_error": ChromaError,
    "onset_only_event_based": partial(
        EventBasedMetric, params={"evaluate_offset": False, "t_collar": 0.2}
    ),
    "segment_based": SegmentBasedMetric,
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

    if "evaluation" not in metadata:
        print(f"Task {task_path.name} has no evaluation config.")
        return

    embedding_type = metadata["embedding_type"]
    predictions: Collection = []
    targets: Collection = []
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
        metric_function = available_metrics[metric](metadata, label_vocab)
        new_results = metric_function(predictions, targets)
        results.update({metric: new_results})

    return results
