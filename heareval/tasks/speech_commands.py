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
from heareval.tasks.util.dataset_builder import DatasetBuilder
from heareval.tasks.util.luigi import (
    PROCESSMETADATACOLS,
    WorkTask,
    filename_to_int_hash,
)


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
                PartitionConfig(name="train", max_files=1000),
                PartitionConfig(name="valid", max_files=100),
                PartitionConfig(name="test", max_files=100),
            ],
        )


class GenerateTrainDataset(WorkTask):
    """
    Silence / background samples in the train / validation sets need to be
    created by slicing up longer background samples into 1sec slices.
    This is the same method used in the TensorFlow dataset generator.
    https://github.com/tensorflow/datasets/blob/79d56e662a15cd11e1fb3b679e0f978c8041566f/tensorflow_datasets/audio/speech_commands.py#L142
    """

    def requires(self):
        # This requires a task called "train" which is an ExtractArchive task.
        raise NotImplementedError

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


class ConfigureProcessMetaData(WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        # This has two requirements: "train" which is a GenerateTrainDataset task
        # and "test" which is an ExtractArchive task.
        raise NotImplementedError()

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
                    .apply(filename_to_int_hash)
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
        process_metadata[PROCESSMETADATACOLS].to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


def main(num_workers: int, sample_rates: List[int]):

    config = SpeechCommandsConfig()
    builder = DatasetBuilder(config)

    # This is a dictionary of the required extraction (untaring) tasks with their
    # required download tasks. This are required be the custom GenerateTrainDataset
    # and ConfigureProcessMetaData tasks.
    download_tasks = builder.download_and_extract_tasks()

    # Run the custom tasks for this dataset to generate samples and configure
    # the metadata files. We instantiate these with the builder to pass in the
    # dynamic download and extract requirements.
    generate_dataset = builder.build_task(
        GenerateTrainDataset, requirements={"train": download_tasks["train"]}
    )
    configure_metadata = builder.build_task(
        ConfigureProcessMetaData,
        requirements={
            "train": generate_dataset,
            "test": download_tasks["test"],
        },
        kwargs={"outfile": "process_metadata.csv"},
    )

    # The remainder of the pipeline is a generic audio pipeline
    # built off of the metadata csv.
    audio_task = builder.prepare_audio_from_metadata_task(
        configure_metadata, sample_rates
    )

    # Run the pipeline
    builder.run(audio_task, num_workers=num_workers)


if __name__ == "__main__":
    main()
