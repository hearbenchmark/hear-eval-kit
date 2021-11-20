"""
Common utils for scoring.
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import sed_eval
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats

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


class MeanAveragePrecision(ScoreFunction):
    """
    Average Precision is calculated in macro mode which calculates AP at a class
    level followed by averaging across the classes
    """

    name = "mAP"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Issue when all ground truths are negative
        # https://github.com/scikit-learn/scikit-learn/issues/8245
        return average_precision_score(targets, predictions, average="macro")


class DPrime(ScoreFunction):
    """
    DPrime is calculated per class followed by averaging across the classes
    Adopted from -
    https://stats.stackexchange.com/questions/492673/understanding-and-implementing-the-dprime-measure-in-python # noqa: E501
    """

    name = "d_prime"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        auc = roc_auc_score(targets, predictions, average=None)
        d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
        # Averaged over the classes
        return np.mean(d_prime)


class LRAP(ScoreFunction):
    """
    Label ranking average precision (LRAP) is the average over each ground
    truth label assigned to each sample, of the ratio of true vs. total
    labels with lower score

    There are two averaging modes:
        * balanced (label weighted ) - per class LRAP, weighted by the overall
            frequency of each class in the ground truth
        * macro - per class LRAP, averaged over the classes as discussed in
            https://github.com/neuralaudio/hear2021-secret-tasks/issues/26

    Adopted from -
    https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=FJv0Rtqfsu3X # noqa: E501
    Author: Dan Ellis dpwe@google.com
    Date: 2019-03-03

    Please refer below link for an overview of LRAP
    https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision # noqa: E501
    """

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        average: str,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param score: Score to use, from the list of overall SED eval scores.
        :param params: Parameters to pass to the scoring function,
                       see inheriting children for details.
        """
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        assert average in ["macro", "balanced"]
        self.average = average

    @staticmethod
    def _one_sample_positive_class_precisions(scores, truth):
        """
        Calculate precisions for each true class for a single sample.
        For instance , If there are multiple true classes for a sample, this
        will return a score for each of the true class, the score being the
        ratio of true vs total labels with lower predicted score than
        prediction score of the label in consideration for the sample

        Args:
            scores: np.array of (num_classes,) giving the individual classifier scores.
            truth: np.array of (num_classes,) bools indicating which classes are true.

        Returns:
            pos_class_indices: np.array of indices of the true classes for this sample.
            pos_class_precisions: np.array of precisions corresponding to each of those
            classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = np.flatnonzero(truth > 0)
        if not len(pos_class_indices):
            return pos_class_indices, np.zeros(0)
        # Rank the labels, according to the predicted score
        retrieved_classes = np.argsort(scores)[::-1]
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = range(num_classes)
        retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        # For each true label, precision is defined as the
        # ratio of true vs total labels below that score
        precision_at_hits = retrieved_cumulative_hits[
            class_rankings[pos_class_indices]
        ] / (1 + class_rankings[pos_class_indices].astype(np.float))
        return pos_class_indices, precision_at_hits

    def calculate_per_class_lrap(self, truth, scores):
        """
        Calculate label-ranking average precision for each class
        Arguments:
            truth: np.array of (num_samples, num_classes) giving boolean ground-truth
            of presence of that class in that sample.
            scores: np.array of (num_samples, num_classes) giving the classifier-under-
            test's real-valued score for each class for each sample.

        Returns:
            per_class_lrap: np.array of (num_classes,) giving the lrap for each
            class.
            weight_per_class: np.array of (num_classes,) giving the prior of each
            class within the truth labels.
        """
        assert truth.shape == scores.shape
        num_samples, num_classes = scores.shape
        precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
        for sample_num in range(num_samples):
            (
                pos_class_indices,
                precision_at_hits,
            ) = self._one_sample_positive_class_precisions(
                scores[sample_num, :], truth[sample_num, :]
            )
            precisions_for_samples_by_classes[
                sample_num, pos_class_indices
            ] = precision_at_hits
        # Determine the number of instances associated with each label
        labels_per_class = np.sum(truth > 0, axis=0)
        # Weight of each class is prior of each class within the true labels
        weight_per_class = labels_per_class / float(np.sum(labels_per_class))
        per_class_lrap = np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(
            1, labels_per_class
        )
        return per_class_lrap, weight_per_class

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        per_class_lrap, weight_per_class = self.calculate_per_class_lrap(
            targets, predictions
        )
        if self.average == "macro":
            return np.mean(per_class_lrap)
        elif self.average == "balanced":
            return np.sum(per_class_lrap * weight_per_class)
        else:
            raise NotImplementedError(
                f"{self.average} mode is not implemented. Only implemented modes are "
                "balanced and macro"
            )


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
    "mAP": MeanAveragePrecision,
    "d_prime": DPrime,
    "lrap": partial(LRAP, name="lrap", average="macro"),
    "lwlrap": partial(LRAP, name="lwlrap", average="balanced"),
}
