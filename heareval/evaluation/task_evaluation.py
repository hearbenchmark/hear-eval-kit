#!/usr/bin/env python3
"""
Compute evaluation scores on a set of predictions for a task.

TODO: This is a start on this file. Still need to add a bunch of scores for the tasks
    including AUC, chroma error for NSYnth pitch, and event-based scores for DCASE.
    Would it make sense to move all the score functions over to a file called scores?
"""

import json
from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
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


# TODO: Add additional scores

available_scores = {"top1_error": top1_error, "macroauc": macroauc}


def task_evaluation(task_path: Path):

    metadata = json.load(task_path.joinpath("task_metadata.json").open())
    label_vocab = pd.read_csv(task_path.joinpath("labelvocabulary.csv"))

    embedding_type = metadata["embedding_type"]

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

    scores = metadata["evaluation"]
    results = {}
    for score in scores:
        print("  -", score)
        new_results = available_scores[score](predictions, targets, label_vocab)
        results.update(new_results)

    return results
