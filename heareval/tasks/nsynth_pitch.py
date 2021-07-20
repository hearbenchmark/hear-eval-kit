#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import os
from pathlib import Path
from functools import partial
import logging

import luigi
import pandas as pd
from slugify import slugify

import heareval.tasks.config.nsynth_pitch as config
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")

# Set the task name for all WorkTasks
luigi_util.WorkTask.task_name = config.TASKNAME


class ExtractArchiveTrain(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.TRAIN_DOWNLOAD_URL, outfile="train-corpus.tar.gz"
            )
        }


class ExtractArchiveValidation(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.VALIDATION_DOWNLOAD_URL, outfile="valid-corpus.tar.gz"
            )
        }


class ExtractArchiveTest(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.TEST_DOWNLOAD_URL, outfile="test-corpus.tar.gz"
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
        metadata = metadata[metadata["pitch"] >= 21]
        metadata = metadata[metadata["pitch"] <= 108]

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
        return {
            "train": SubsamplePartition(
                partition="train", max_files=config.MAX_TRAIN_FILES
            ),
            "test": SubsamplePartition(
                partition="test", max_files=config.MAX_TEST_FILES
            ),
            "validation": SubsamplePartition(
                partition="valid", max_files=config.MAX_VAL_FILES
            ),
        }

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
    def requires(self):
        # Will copy the resampled data and the traintestmeta and the vocabmeta
        return {
            "resample": [
                ResampleSubCorpus(sr, partition)
                for sr in config.SAMPLE_RATES
                for partition in ["train", "test", "valid"]
            ],
            "traintestmeta": SplitTrainTestMetadata(),
            "vocabmeta": MetadataVocabulary(),
        }


def main():
    luigi_util.ensure_dir("_workdir")
    luigi.build(
        [FinalizeCorpus()],
        workers=config.NUM_WORKERS,
        local_scheduler=True,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
