#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from functools import partial
from glob import glob
from pathlib import Path

import heareval.tasks.config.speech_commands as config
import luigi
import numpy as np
import pandas as pd
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
    SubsampleCorpus,
    WorkTask,
    ensure_dir,
    filename_to_int_hash,
    slugify_file_name,
    which_set,
)

# Set the task name for all WorkTasks
WorkTask.task_name = config.TASKNAME

BACKGROUND_NOISE = "_background_noise_"


class ExtractArchiveTrain(ExtractArchive):
    def requires(self):
        return {
            "download": DownloadCorpus(
                url=config.DOWNLOAD_URL, outfile="train-corpus.tar.gz"
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

    def requires(self):
        return {
            "train": ExtractArchiveTrain(infile="train-corpus.tar.gz"),
            "test": ExtractArchiveTest(infile="test-corpus.tar.gz"),
        }

    def run(self):

        train_path = Path(self.requires()["train"].workdir)

        # List of all relative paths to the testing and validation files
        with open(os.path.join(train_path, "testing_list.txt"), "r") as fp:
            train_test_paths = fp.read().strip().splitlines()

        with open(os.path.join(train_path, "validation_list.txt"), "r") as fp:
            validation_paths = fp.read().strip().splitlines()

        all_audio = [str(p.relative_to(train_path)) for p in train_path.glob("*/*.wav")]

        # Need to manually add a background noise file into the validation set.
        validation_paths.append(os.path.join(BACKGROUND_NOISE, "running_tap.wav"))

        # The training set is files that are not in either the train-test or validation.
        train_paths = set(all_audio) - set(train_test_paths) - set(validation_paths)

        # The testing set is all the files in the separate testing download
        # test_path = Path(self.requires()["test"].workdir)
        # test_paths = [p.relative_to(test_path) for p in test_path.glob("*/*.wav")]

        process_metadata = pd.DataFrame(train_paths, columns=["relpath"])
        process_metadata["slug"] = process_metadata["relpath"].apply(slugify_file_name)

        # Save the process metadata
        process_metadata.to_csv(
            os.path.join(self.workdir, "process_metadata.csv"),
            columns=["relpath", "slug"],
            header=False,
            index=False,
        )

        # Get relative path of the audio files
        # This file can also be built with the metadata file for the dataset
        # in case the metadata is provided
        process_metadata = pd.DataFrame(
            glob(os.path.join(self.requires().workdir, "[!_]*/*.wav")),
            columns=["relpath"],
        )

        # Get unique slug for each file
        # In this case the base name is not excluded since same person have made dataset
        # for different category
        # This is something which is data dependent and should be handled here
        process_metadata["slug"] = (
            process_metadata["relpath"]
            # Dont exclude the base name. Apply sluggify on all
            .apply(partial(os.path.relpath, start=self.requires().workdir)).apply(
                slugify_file_name
            )
        )

        # Hash the field id rather than the full path.
        # This hashing is specific to the dataset and should be done here
        # In this case we take the slug and remove the -nohash- as described
        # This nohash removal allows for the speech of similar person to be in the
        # same dataset. Such type of data specific requirements might be there.
        # in the readme of google speech commands. we want to keep similar people
        # in the same group - test or train or val
        process_metadata["filename_hash"] = (
            process_metadata["slug"]
            .apply(lambda relpath: re.sub(r"-nohash-.*$", "", relpath))
            .apply(filename_to_int_hash)
        )

        # Get the data partition either from the input file or by doing the hash
        process_metadata["partition"] = process_metadata["filename_hash"].apply(
            partial(which_set, validation_percentage=0.0, testing_percentage=10.0)
        )

        # Get label for the data from anywhere.
        # In this case it is the folder name
        process_metadata["label"] = process_metadata["relpath"].apply(
            lambda relative_path: os.path.basename(os.path.dirname(relative_path))
        )

        # Save the process metadata
        process_metadata.to_csv(
            os.path.join(self.workdir, "process_metadata.csv"),
            columns=PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        with self.output().open("w") as _:
            pass


class SubsampleCorpus(SubsampleCorpus):
    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {"meta": ConfigureProcessMetaData()}


class MonoWavTrimCorpus(MonoWavTrimCorpus):
    def requires(self):
        return {
            "corpus": SubsampleCorpus(max_file_per_corpus=config.MAX_FILES_PER_CORPUS)
        }


class SplitTrainTestCorpus(SplitTrainTestCorpus):
    def requires(self):
        # The metadata helps in provide the partition type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(min_sample_length=config.SAMPLE_LENGTH_SECONDS),
            "meta": ConfigureProcessMetaData(),
        }


class SplitTrainTestMetadata(SplitTrainTestMetadata):
    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(),
            "meta": ConfigureProcessMetaData(),
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
                for partition in ["train", "test", "val"]
            ],
            "traintestmeta": SplitTrainTestMetadata(),
            "vocabmeta": MetadataVocabulary(),
        }


def main():
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    ensure_dir("_workdir")

    luigi.build(
        [FinalizeCorpus()],
        workers=config.NUM_WORKERS,
        local_scheduler=True,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
