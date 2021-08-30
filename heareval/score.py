"""
Common utils for scoring.
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import sed_eval
import torch

# Can we get away with not using DCase for every event-based evaluation??
from dcase_util.containers import MetaDataContainer


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros((num_labels,), dtype=torch.float)
    else:
        binary_labels = torch.zeros((num_labels,)).scatter(0, torch.tensor(label), 1.0)

    # Validate the binary vector we just created
    assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
    return binary_labels


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, predictions: Any, targets: Any, **kwargs) -> float:
        """
        Compute the score based on the predictions and targets. Returns the score.
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)


class ChromaAccuracy(ScoreFunction):
    """
    Score specifically for pitch detection -- converts all pitches to chroma first.
    This score ignores octave errors in pitch classification.
    """

    name = "chroma_acc"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        return correct / len(targets)


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        score: str,
        params: Dict = None,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param score: Score to use, from the list of overall SED eval scores.
        :param params: Parameters to pass to the scoring function,
                       see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.score = score
        self.params = params
        assert self.score_class is not None

    def __call__(self, predictions: Dict, targets: Dict, **kwargs):
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # This (and segment_based_scores) return a pretty large selection of scores.
        overall_scores = scores.results_overall_metrics()
        # Keep the specific score we want
        return overall_scores[self.score][self.score]

    @staticmethod
    def sed_eval_event_container(
        x: Dict[str, List[Dict[str, Any]]]
    ) -> MetaDataContainer:
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


class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


available_scores: Dict[str, Callable] = {
    "top1_acc": Top1Accuracy,
    "pitch_acc": partial(Top1Accuracy, name="pitch_acc"),
    "chroma_acc": ChromaAccuracy,
    # https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html
    "event_onset_200ms_fms": partial(
        EventBasedScore,
        name="event_onset_200ms_fms",
        score="f_measure",
        params={
            "evaluate_onset": True,
            "evaluate_offset": False,
            "t_collar": 0.2,
            "percentage_of_length": 0.5,
        },
    ),
    "segment_1s_er": partial(
        SegmentBasedScore,
        name="segment_1s_er",
        score="error_rate",
        params={"time_resolution": 1.0},
        maximize=False,
    ),
}
