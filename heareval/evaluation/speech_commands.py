#!/usr/bin/env python3

"""
Run evaluation on google speech command predictions
"""
from typing import Dict
import numpy as np
import sys
import argparse
import csv
import pandas as pd


def generate_random_predictions(
    num_labels: int, audio_length: flaot = 1.0, hop_size: float = 33.0
) -> np.ndarray:
    """
    Generate a set of random predictions for testing the evaluation

    Receives the number of labels, then produces a random prediction based on
    those labels at regular intervals (hop_size in ms) to mimic an audio file
    of length audio_length in seconds.
    """
    hop_size = hop_size / 1000.0
    num_hops = int(audio_length // hop_size)

    # Values in range[0.0, 1.0) represent probabilities for each label
    # at every frame
    return np.random.random((num_hops, num_labels))


def evaluate_framewise_predictions(
    pred: Dict[str, np.ndarry], truth: Dict[str, int]
) -> float:
    """
    Evaluate the framewise predictions using top-1 error
    """
    correct = 0
    for filename, label in truth.items():
        # Average all the predictions over time and select the one
        # that received the highest probability
        predicted_class = np.argmax(pred[filename].mean(axis=0))
        if predicted_class == label:
            correct += 1

    top_1_error = correct / len(pred)
    return top_1_error


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("truth", help="Ground truth csv", type=str)

    args = parser.parse_args(arguments)

    # Load in truth file csv
    truth = pd.read_csv(args.truth, header=0)

    # Get all the labels
    labels = set()
    for item in truth.itertuples():
        labels.add(item[2])

    # Keep labels ordered so we can associate integers with them
    labels = sorted(list(labels))

    print(
        f"Performing evaluation on {len(truth)} predictions"
        f" with {len(labels)} ground truth labels"
    )

    # We will eventually load in the prediction file, but here we will generate
    # random framewise predictions for each audio file
    ground_truth = {}
    predictions = {}
    for item in truth.itertuples():
        filename = item[1]
        framewise_predictions = generate_random_predictions(len(labels))
        predictions[filename] = framewise_predictions

        # Associated integer label for the ground truth
        ground_truth[filename] = labels.index(item[2])

    # Perform evaluation
    error = evaluate_framewise_predictions(predictions, ground_truth)
    print("Top-1 Error for predictions:", error)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
