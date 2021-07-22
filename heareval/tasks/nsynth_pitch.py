#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import os
from pathlib import Path
from functools import partial
import logging
from typing import List

import luigi
import pandas as pd
from slugify import slugify

import heareval.tasks.util.luigi as luigi_util
from heareval.tasks.dataset_config import (
    PartitionedDatasetConfig,
    PartitionConfig,
)

logger = logging.getLogger("luigi-interface")


# Dataset configuration
class NSynthPitchConfig(PartitionedDatasetConfig):
    def __init__(self):
        super().__init__(
            task_name="nsynth-pitch",
            version="v2.2.3",
            download_urls={
                "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa: E501
                "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa: E501
                "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa: E501
            },
            # All samples will be trimmed / padded to this length
            sample_duration=4.0,
            # Pre-defined partitions in the dataset. Number of files in each split is
            # train: 85,111; valid: 10,102; test: 4890. These values will be a bit less
            # after filter the pitches to be only within the piano range.
            # To subsample a partition, set the max_files to an integer.
            # TODO: Should we subsample NSynth?
            partitions=[
                PartitionConfig(name="train", max_files=10000),
                PartitionConfig(name="valid", max_files=1000),
                PartitionConfig(name="test", max_files=None),
            ],
        )
        # We only include pitches that are on a standard 88-key MIDI piano
        self.pitch_range = (21, 108)


# Set the task name for all WorkTasks
config = NSynthPitchConfig()
luigi_util.WorkTask.task_name = config.versioned_task_name


class ExtractArchiveTrain(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.download_urls["train"], outfile="train-corpus.tar.gz"
            )
        }


class ExtractArchiveValidation(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.download_urls["valid"], outfile="valid-corpus.tar.gz"
            )
        }


class ExtractArchiveTest(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.download_urls["test"], outfile="test-corpus.tar.gz"
            )
        }


class ConfigureProcessMetaData(luigi_util.WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        return {
            "train": ExtractArchiveTrain(infile="train-corpus.tar.gz"),
            "valid": ExtractArchiveValidation(infile="valid-corpus.tar.gz"),
            "test": ExtractArchiveTest(infile="test-corpus.tar.gz"),
        }

    @staticmethod
    def get_rel_path(root: Path, item: pd.DataFrame) -> str:
        # Creates the audio relative path for a dataframe item
        audio_path = root.joinpath("audio")
        filename = f"{item}.wav"
        return audio_path.joinpath(filename)

    @staticmethod
    def slugify_file_name(filename: str) -> str:
        return f"{slugify(filename)}.wav"

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(f"nsynth-{split}")

        metadata = pd.read_json(split_path.joinpath("examples.json"), orient="index")

        # Filter out pitches that are not within the range of a standard piano
        metadata = metadata[metadata["pitch"] >= config.pitch_range[0]]
        metadata = metadata[metadata["pitch"] <= config.pitch_range[1]]

        metadata = metadata.assign(label=lambda df: df["pitch"])
        metadata = metadata.assign(
            relpath=lambda df: df["note_str"].apply(
                partial(self.get_rel_path, split_path)
            )
        )
        metadata = metadata.assign(
            slug=lambda df: df["note_str"].apply(self.slugify_file_name)
        )
        metadata = metadata.assign(partition=lambda df: split)
        metadata = metadata.assign(
            filename_hash=lambda df: df["slug"].apply(luigi_util.filename_to_int_hash)
        )

        return metadata[luigi_util.PROCESSMETADATACOLS]

    def run(self):

        # Get metadata for each of the data splits
        process_metadata = pd.concat(
            [self.get_split_metadata(split) for split in self.requires()]
        )

        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=luigi_util.PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


class SubsamplePartition(luigi_util.SubsamplePartition):
    """
    A subsampler that acts on a specific partition.
    All instances of this will depend on the combined process metadata csv.
    """

    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {
            "meta": ConfigureProcessMetaData(outfile="process_metadata.csv"),
        }


class SubsamplePartitions(luigi_util.WorkTask):
    """
    Aggregates subsampling of all the partitions into a single task as dependencies.
    All the subsampled files are stored in the requires workdir, so we just link to
    that since there aren't any real outputs associated with this task.
    This is a bit of a hack -- but it allows us to avoid rewriting
    the Subsample task as well as take advantage of Luigi concurrency.
    """

    def requires(self):
        # Perform subsampling on each partition independently
        subsample_partitions = {
            partition.name: SubsamplePartition(
                partition=partition.name, max_files=partition.max_files
            )
            for partition in config.partitions
        }
        return subsample_partitions

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        workdir.symlink_to(Path(self.requires()["train"].workdir).absolute())
        self.mark_complete()


class SplitTrainTestCorpus(luigi_util.SplitTrainTestCorpus):
    def requires(self):
        # The metadata helps in provide the partition type for each
        # audio file
        return {
            "corpus": SubsamplePartitions(),
            "meta": ConfigureProcessMetaData(outfile="process_metadata.csv"),
        }


class SplitTrainTestMetadata(luigi_util.SplitTrainTestMetadata):
    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(),
            "meta": ConfigureProcessMetaData(outfile="process_metadata.csv"),
        }


class MetadataVocabulary(luigi_util.MetadataVocabulary):
    def requires(self):
        # Depends only on the train test metadata
        return {"traintestmeta": SplitTrainTestMetadata()}


class ResampleSubCorpus(luigi_util.ResampleSubCorpus):
    def requires(self):
        # Requires the train test corpus and will take in
        # parameter for which partition and sr the resampling
        # has to be done
        return {"traintestcorpus": SplitTrainTestCorpus()}


class FinalizeCorpus(luigi_util.FinalizeCorpus):

    sample_rates = luigi.ListParameter()

    def requires(self):
        # Will copy the resampled data and the traintestmeta and the vocabmeta
        return {
            "resample": [
                ResampleSubCorpus(sr, partition)
                for sr in self.sample_rates
                for partition in ["train", "test", "valid"]
            ],
            "traintestmeta": SplitTrainTestMetadata(),
            "vocabmeta": MetadataVocabulary(),
        }


def main(num_workers: int, sample_rates: List[int]):
    luigi_util.ensure_dir("_workdir")
    luigi.build(
        [FinalizeCorpus(sample_rates=sample_rates)],
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
    )
