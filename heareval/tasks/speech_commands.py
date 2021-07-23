#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from pathlib import Path
from typing import List

import luigi
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from slugify import slugify

from heareval.tasks.dataset_config import (
    PartitionedDatasetConfig,
    PartitionConfig,
)
import heareval.tasks.util.luigi as luigi_util


WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
BACKGROUND_NOISE = "_background_noise_"
UNKNOWN = "_unknown_"
SILENCE = "_silence_"


class SpeechCommandsConfig(PartitionedDatasetConfig):
    def __init__(self):
        super().__init__(
            task_name="speech_commands",
            version="v0.0.2",
            download_urls={
                "train": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",  # noqa: E501
                "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",  # noqa: E501
            },
            # All samples will be trimmed / padded to this length
            sample_duration=1.0,
            # Pre-defined partitions in the dataset. Number of files in each split is
            # train: 85,111; valid: 10,102; test: 4890.
            # To subsample a partition, set the max_files to an integer.
            partitions=[
                PartitionConfig(name="train", max_files=None),
                PartitionConfig(name="valid", max_files=None),
                PartitionConfig(name="test", max_files=None),
            ],
        )


config = SpeechCommandsConfig()
luigi_util.WorkTask.task_name = config.versioned_task_name


class ExtractArchiveTrain(luigi_util.ExtractArchive):

    download = luigi.TaskParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )

    def requires(self):
        return {
            "download": self.download,
            # "download": luigi_util.DownloadCorpus(
            #     url=config.download_urls["train"], outfile="train-corpus.tar.gz"
            # )
        }


class ExtractArchiveTest(luigi_util.ExtractArchive):
    def requires(self):
        return {
            "download": luigi_util.DownloadCorpus(
                url=config.download_urls["test"], outfile="test-corpus.tar.gz"
            )
        }


class GenerateTrainDataset(luigi_util.WorkTask):
    """
    Silence / background samples in the train / validation sets need to be
    created by slicing up longer background samples into 1sec slices.
    This is the same method used in the TensorFlow dataset generator.
    https://github.com/tensorflow/datasets/blob/79d56e662a15cd11e1fb3b679e0f978c8041566f/tensorflow_datasets/audio/speech_commands.py#L142
    """

    def requires(self):
        return {"train": ExtractArchiveTrain(infile="train-corpus.tar.gz")}

    def run(self):
        train_path = Path(self.requires()["train"].workdir)
        background_audio = list(train_path.glob(f"{BACKGROUND_NOISE}/*.wav"))

        # Read all the background audio files and split into 1 second segments,
        # save all the segments into a folder called _silence_
        silence_dir = os.path.join(self.workdir, SILENCE)
        os.makedirs(silence_dir, exist_ok=True)

        print("Generating silence files from background sounds ...")
        for audio_path in tqdm(background_audio):
            audio, sr = sf.read(audio_path)

            basename = os.path.basename(audio_path)
            name, ext = os.path.splitext(basename)

            for start in range(0, len(audio) - sr, sr // 2):
                audio_segment = audio[start : start + sr]
                filename = f"{name}-{start}{ext}"
                filename = os.path.join(silence_dir, filename)
                sf.write(filename, audio_segment, sr)

        # We'll also create symlinks for the dataset here too to make the next
        # stage of splitting into training and validation files easier.
        for file_obj in train_path.iterdir():
            if file_obj.is_dir() and file_obj.name != BACKGROUND_NOISE:
                linked_folder = Path(os.path.join(self.workdir, file_obj.name))
                linked_folder.unlink(missing_ok=True)
                linked_folder.symlink_to(file_obj.absolute(), target_is_directory=True)

            # Also need the testing and validation splits
            if file_obj.name in ["testing_list.txt", "validation_list.txt"]:
                linked_file = Path(os.path.join(self.workdir, file_obj.name))
                linked_file.unlink(missing_ok=True)
                linked_file.symlink_to(file_obj.absolute())

        self.mark_complete()


class ConfigureProcessMetaData(luigi_util.WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        return {
            "train": GenerateTrainDataset(),
            "test": ExtractArchiveTest(infile="test-corpus.tar.gz"),
        }

    @staticmethod
    def apply_label(relative_path):
        label = os.path.basename(os.path.dirname(relative_path))
        if label not in WORDS and label != SILENCE:
            label = UNKNOWN
        return label

    @staticmethod
    def slugify_file_name(relative_path):
        folder = os.path.basename(os.path.dirname(relative_path))
        basename = os.path.basename(relative_path)
        name, ext = os.path.splitext(basename)
        return f"{slugify(os.path.join(folder, name))}{ext}"

    def get_split_paths(self):

        # Test files
        test_path = Path(self.requires()["test"].workdir)
        test_df = pd.DataFrame(test_path.glob("*/*.wav"), columns=["relpath"]).assign(
            partition=lambda df: "test"
        )

        # All silence paths to add to the train and validation
        train_path = Path(self.requires()["train"].workdir)
        all_silence = list(train_path.glob(f"{SILENCE}/*.wav"))

        # Validation files
        with open(os.path.join(train_path, "validation_list.txt"), "r") as fp:
            validation_paths = fp.read().strip().splitlines()
        validation_rel_paths = [os.path.join(train_path, p) for p in validation_paths]
        val_silence = list(train_path.glob(f"{SILENCE}/running_tap*.wav"))
        validation_rel_paths.extend(val_silence)
        validation_df = pd.DataFrame(validation_rel_paths, columns=["relpath"]).assign(
            partition=lambda df: "valid"
        )

        # Train files
        with open(os.path.join(train_path, "testing_list.txt"), "r") as fp:
            train_test_paths = fp.read().strip().splitlines()
        audio_paths = [
            str(p.relative_to(train_path)) for p in train_path.glob("[!_]*/*.wav")
        ]

        train_paths = list(
            set(audio_paths) - set(train_test_paths) - set(validation_paths)
        )
        train_rel_paths = [os.path.join(train_path, p) for p in train_paths]

        train_silence = list(set(all_silence) - set(val_silence))
        train_rel_paths.extend(train_silence)
        train_df = pd.DataFrame(train_rel_paths, columns=["relpath"]).assign(
            partition=lambda df: "train"
        )
        assert len(train_df.merge(validation_df, on="relpath")) == 0

        return pd.concat([test_df, validation_df, train_df])

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
                    .apply(luigi_util.filename_to_int_hash)
                )
            )
            # Get label for the data from anywhere.
            # In this case it is the folder name
            .assign(label=lambda df: df["relpath"].apply(self.apply_label))
        )

        return process_metadata

    def run(self):
        process_metadata = self.get_split_paths()
        process_metadata = self.get_metadata_attrs(process_metadata)

        # Save the process metadata
        process_metadata[luigi_util.PROCESSMETADATACOLS].to_csv(
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


class MonoWavTrimCorpus(luigi_util.MonoWavTrimCorpus):
    def requires(self):
        return {"corpus": SubsamplePartitions()}


class SplitTrainTestCorpus(luigi_util.SplitTrainTestCorpus):
    def requires(self):
        # The metadata helps in provide the partition type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(duration=config.sample_duration),
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


class FinalizeCorpus(luigi_util.WorkTask):

    sample_rates = luigi.ListParameter()
    next_task = luigi.TaskParameter()

    def requires(self):
        # Will copy the resampled data and the traintestmeta and the vocabmeta
        return {
            "resample": [
                ResampleSubCorpus(sr=sr, partition=partition)
                for sr in self.sample_rates
                for partition in ["train", "test", "valid"]
            ],
            "traintestmeta": SplitTrainTestMetadata(),
            "vocabmeta": MetadataVocabulary(),
        }


def main(num_workers: int, sample_rates: List[int]):
    luigi_util.ensure_dir("_workdir")

    download = luigi_util.DownloadCorpus(
        url=config.download_urls["train"], outfile="train-corpus.tar.gz"
    )
    extract = ExtractArchiveTrain(infile="train-corpus.tar.gz", download=download)

    luigi.build(
        [extract],
        # [FinalizeCorpus(sample_rates=sample_rates, task_string=config.versioned_task_name, next_task=GenerateTrainDataset)],
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
    )
