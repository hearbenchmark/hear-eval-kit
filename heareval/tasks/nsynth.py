#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import os
import re
from pathlib import Path

import luigi
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from slugify import slugify

import heareval.tasks.config.nsynth as config
from heareval.tasks.util.luigi import (
    PROCESSMETADATACOLS,
    DownloadCorpus,
    ExtractArchive,
    FinalizeCorpus,
    MetadataVocabulary,
    MonoWavTrimCorpus,
    ResampleSubCorpus,
    SplitTrainTestCorpus,
    SplitTrainTestMetadata,
    SubsamplePartition,
    WorkTask,
    ensure_dir,
    filename_to_int_hash,
)

# Set the task name for all WorkTasks
WorkTask.task_name = config.TASKNAME


class ExtractArchiveTrain(ExtractArchive):
    def requires(self):
        return {
            "download": DownloadCorpus(
                url=config.TRAIN_DOWNLOAD_URL, outfile="train-corpus.tar.gz"
            )
        }


class ExtractArchiveValidation(ExtractArchive):
    def requires(self):
        return {
            "download": DownloadCorpus(
                url=config.VALIDATION_DOWNLOAD_URL, outfile="valid-corpus.tar.gz"
            )
        }


class ExtractArchiveTest(ExtractArchive):
    def requires(self):
        return {
            "download": DownloadCorpus(
                url=config.TEST_DOWNLOAD_URL, outfile="test-corpus.tar.gz"
            )
        }


class ConfigureProcessMetaData(WorkTask):
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
    def apply_label(relative_path):
        label = os.path.basename(os.path.dirname(relative_path))
        # if label not in WORDS and label != SILENCE:
        #     label = UNKNOWN
        return label

    @staticmethod
    def slugify_file_name(relative_path):
        folder = os.path.basename(os.path.dirname(relative_path))
        basename = os.path.basename(relative_path)
        name, ext = os.path.splitext(basename)
        return f"{slugify(os.path.join(folder, name))}{ext}"

    def get_split_paths(self):
        return None

    def get_metadata_attrs(self, process_metadata):
        process_metadata = (
            process_metadata
            # Create a unique slug for each file. We include the folder with the class
            # name b/c the base filenames may not be unique.
            .assign(slug=lambda df: df["relpath"].apply(self.slugify_file_name))
            # Hash the field id rather than the full path.
            # This hashing is specific to the dataset and should be done here
            # In this case we take the slug and remove the -nohash- as described
            # This nohash removal allows for the speech of similar person to be in the
            # same dataset. Such type of data specific requirements might be there.
            # in the readme of google speech commands. we want to keep similar people
            # in the same group - test or train or val
            .assign(
                filename_hash=lambda df: (
                    df["slug"]
                    .apply(lambda relpath: re.sub(r"-nohash-.*$", "", relpath))
                    .apply(filename_to_int_hash)
                )
            )
            # Get label for the data from anywhere.
            # In this case it is the folder name
            .assign(label=lambda df: df["relpath"].apply(self.apply_label))
        )

        return process_metadata

    def run(self):

        assert False

        self.mark_complete()


class SubsamplePartition(SubsamplePartition):
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


class SubsamplePartitions(WorkTask):
    """
    Aggregates subsampling of all the partitions into a single task as dependencies.
    All the subsampled files are stored in the requires workdir, so we just link to
    that since there aren't any real outputs associated with this task.
    This is a bit of a hack -- but it allows us to avoid rewriting
    the Subsample task as well as take advantage of Luigi concurrency.
    """

    def requires(self):
        # Perform subsampling on each partition independently
        return {
            "train": SubsamplePartition(
                partition="train", max_files=config.MAX_TRAIN_FILES
            ),
            "test": SubsamplePartition(
                partition="test", max_files=config.MAX_TEST_FILES
            ),
            "validation": SubsamplePartition(
                partition="validation", max_files=config.MAX_VAL_FILES
            ),
        }

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        workdir.symlink_to(Path(self.requires()["train"].workdir).absolute())
        self.mark_complete()


class MonoWavTrimCorpus(MonoWavTrimCorpus):
    def requires(self):
        return {"corpus": SubsamplePartitions()}


class SplitTrainTestCorpus(SplitTrainTestCorpus):
    def requires(self):
        # The metadata helps in provide the partition type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(duration=config.SAMPLE_LENGTH_SECONDS),
            "meta": ConfigureProcessMetaData(outfile="process_metadata.csv"),
        }


class SplitTrainTestMetadata(SplitTrainTestMetadata):
    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(),
            "meta": ConfigureProcessMetaData(outfile="process_metadata.csv"),
        }


class MetadataVocabulary(MetadataVocabulary):
    def requires(self):
        # Depends only on the train test metadata
        return {"traintestmeta": SplitTrainTestMetadata()}


class ResampleSubCorpus(ResampleSubCorpus):
    def requires(self):
        # Requires the train test corpus and will take in
        # parameter for which partition and sr the resampling
        # has to be done
        return {"traintestcorpus": SplitTrainTestCorpus()}


class FinalizeCorpus(FinalizeCorpus):
    def requires(self):
        # Will copy the resampled data and the traintestmeta and the vocabmeta
        return {
            "resample": [
                ResampleSubCorpus(sr, partition)
                for sr in config.SAMPLE_RATES
                for partition in ["train", "test", "validation"]
            ],
            "traintestmeta": SplitTrainTestMetadata(),
            "vocabmeta": MetadataVocabulary(),
        }


def main():
    ensure_dir("_workdir")
    luigi.build(
        [FinalizeCorpus()],
        workers=config.NUM_WORKERS,
        local_scheduler=True,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
