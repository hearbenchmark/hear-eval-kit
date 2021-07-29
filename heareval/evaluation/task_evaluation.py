#!/usr/bin/env python3
"""
Compute evaluation metrics on a set of predictions for a task.

TODO: This is a start on this file. Still need to add a bunch of metrics for the tasks
    including AUC, chroma error for NSYnth pitch, and event-based metrics for DCASE.
    Would it make sense to move all the metric functions over to a file called metrics?
"""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
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


available_metrics = {
    "top1_error": Top1Error,
    "macroauc": MacroAUC,
    "chroma_error": ChromaError,
}


def task_evaluation(task_path: Path):

    metadata = json.load(task_path.joinpath("task_metadata.json").open())
    label_vocab = pd.read_csv(task_path.joinpath("labelvocabulary.csv"))

    if "evaluation" not in metadata:
        print(f"Task {task_path.name} has no evaluation config.")
        return

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

    # TODO: Check shape of predictions vs label_vocab

    # What other types could we receive as targets?
    assert isinstance(targets, list)

    # Make sure we have the same number of predictions as targets
    assert len(predictions) == len(targets)

    metrics = metadata["evaluation"]
    results = {}
    for metric in metrics:
        print("  -", metric)
        metric_function = available_metrics[metric](metadata, label_vocab)
        new_results = metric_function(predictions, targets)
        results.update(new_results)

    return results
