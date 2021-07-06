#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from functools import partial
from glob import glob

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
import tensorflow as tf
import tensorflow_datasets as tfds

# Set the task name for all WorkTasks
WorkTask.task_name = config.TASKNAME


class LoadTensorFlowDataset(WorkTask):

    dataset = luigi.Parameter()
    version = luigi.Parameter()

    def run(self):
        # This downloads and saves the full dataset as tfrecords. The nice thing about
        # using this method is that it sets up the train/val/test splits for us
        # according to the pre-defined method. Also, it generates the _silence_ examples
        # too which we weren't doing before.
        builder = tfds.builder(
            self.dataset, version=self.version, data_dir=self.workdir
        )
        builder.download_and_prepare()
        with self.output().open("w") as _:
            pass

    @property
    def stage_number(self) -> int:
        return 0


class ExportTFDS(WorkTask):
    def requires(self):
        return {
            "download": LoadTensorFlowDataset("speech_commands", "0.0.2"),
        }

    def run(self):
        download = self.requires()["download"]
        builder = tfds.builder(download.dataset, data_dir=download.workdir)

        train = builder.as_dataset(split="train", shuffle_files=False)
        val = builder.as_dataset(split="validation", shuffle_files=False)
        test = builder.as_dataset(split="test", shuffle_files=False)

        assert isinstance(train, tf.data.Dataset)

        # Sample rate of all the audio files
        sample_rate = builder._info().features["audio"].sample_rate

        # We can also get the text labels for each class
        labels = builder._info().features["label"]
        print(labels.num_classes)
        print(labels.names)

        for item in train:
            # We can save all the audio files for the pre-defined splits as wav files.
            audio = item["audio"]
            label = item["label"]

        # Do the same for the validation and test sets

        # TODO: not so sure about how subsampling works after this?


class ExtractArchive(ExtractArchive):
    def requires(self):
        return {
            "download": DownloadCorpus(url=config.DOWNLOAD_URL, outfile="corpus.tar.gz")
        }


class ConfigureProcessMetaData(WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    def requires(self):
        return ExtractArchive(infile="corpus.tar.gz")

    def run(self):
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
        [ExportTFDS()],
        workers=config.NUM_WORKERS,
        local_scheduler=True,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
