import os
import csv
import numpy as np
from numpy.lib.function_base import append
from sklearn.metrics import roc_auc_score, top_k_accuracy_score
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


def csv_to_dict(file_path: str, cast=float) -> dict:
    """Read CSV file (without header) from disk into a dictionary.

    Args:
        file_path: (str): Path to CSV file.
        row_type (type): Cast CSV values.
    Returns:
        d (dict): Dictionary with keys containing first column,
        and values corresponding to all other columns.

    """
    with open(file_path, "r") as fp:
        reader = csv.reader(fp)
        d = {}
        for row in reader:
            vals = []
            for ridx, val in enumerate(row):
                if ridx == 0:
                    filename = val
                else:
                    vals.append(cast(val))

            if len(vals) > 1:
                row_values = np.array(vals)
            else:
                row_values = vals[0]

            d[filename] = row_values

    return d


def str_label_to_int(d_str: dict, labels: dict) -> dict:
    """Convert dictionary entries (values) based on matching integer definitions.

    Args:
        d_str (dict): String-based labeled items.
        labels (dict): Label vocabulary.

    Returns:
        d_int (dict): Labeled items with interger labels.
    """

    d_int = {}
    for key, val in d_str.items():
        d_int[key] = labels[val]

    return d_int


def align_predictions(test, pred) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over predictions to find correponding ground truth.

    Args:
        test (dict):
        pred (dict):

    Returns:
        y_true (array): Target of shape (n_samples,) or (n_samples, n_classes)
        y_pred (array): Predicted scores of shape (n_samples,) or (n_samples, n_classes)

    This is only set of for binary and multiclass.
    In the case of multilabel, y_true must be of shape (n_samples, n_classes).

    """
    y_pred = []
    y_true = []

    for file_id, scores in pred.items():
        y_pred.append(scores)
        y_true.append(test[file_id])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_true, y_pred


def evaluate(
    test_csv_file_path: str,
    pred_csv_file_path: str,
    labelvocabulary_csv_filepath: str,
    task_type: str = "classification",
) -> dict:
    """Compute evaluation metrics.

    Args:
        test_csv_file_path (str): CSV file containing correct labels.
        pred_csv_file_path (str): CSV file containing scores produced by a model.
        labelvocabulary_csv_filepath (str): CSV file containing integer values for string labels.
        tast_type (str, optional): One of the five task types.
            ["classification", "tagging", "temporal", "ranking", "jnd"]

    Returns
        metrics (dict): Evaluation metrics.

    """

    # check if files exist
    if not os.path.isfile(test_csv_file_path):
        raise RuntimeError(f"test_csv_file_path: {test_csv_file_path} does not exist.")

    if not os.path.isfile(pred_csv_file_path):
        raise RuntimeError(f"pred_csv_file_path: {pred_csv_file_path} does not exist.")

    if not os.path.isfile(labelvocabulary_csv_filepath):
        raise RuntimeError(
            f"labelvocabulary_csv_filepath: {labelvocabulary_csv_filepath} does not exist."
        )

    # check if for valid task
    if task_type not in ["classification", "tagging", "temporal", "ranking", "jnd"]:
        raise ValueError(f"Invalid task_type: {task_type}.")

    # Note: we don't assume that both files are ordered
    test = csv_to_dict(test_csv_file_path, cast=str)
    pred = csv_to_dict(pred_csv_file_path)
    labels = csv_to_dict(labelvocabulary_csv_filepath, cast=int)

    # parse test dict to convert string categories to ints
    test = str_label_to_int(test, labels)

    # iterate over elements in predictions to create aligned arrays
    y_true, y_score = align_predictions(test, pred)

    # compute metrics based on scores and ground truth
    auc = roc_auc_score(y_true, y_score, multi_class="ovr")
    top_k = top_k_accuracy_score(y_true, y_score, k=2)

    metrics = {"auc": auc, "top_k": top_k}

    return metrics


if __name__ == "__main__":
    metrics = evaluate(
        "test.csv",
        "predicted-test.csv",
        "_workdir/08-MetadataVocabulary/labelvocabulary.csv",
    )

    print(metrics)
