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

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util

WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
BACKGROUND_NOISE = "_background_noise_"
UNKNOWN = "_unknown_"
SILENCE = "_silence_"


config = {
    "task_name": "speech_commands",
    "version": "v0.0.2",
    "task_type": "scene_labeling",
    "download_urls": {
        "train": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",  # noqa: E501
        "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",  # noqa: E501
    },
    "sample_duration": 1.0,
    "splits": [
        {"name": "train", "max_files": 100},
        {"name": "test", "max_files": 100},
        {"name": "valid", "max_files": 100},
    ],
}


class GenerateTrainDataset(luigi_util.WorkTask):
    """
    Silence / background samples in the train / validation sets need to be
    created by slicing up longer background samples into 1sec slices.
    This is the same method used in the TensorFlow dataset generator.
    https://github.com/tensorflow/datasets/blob/79d56e662a15cd11e1fb3b679e0f978c8041566f/tensorflow_datasets/audio/speech_commands.py#L142
    """

    # Requires an extracted dataset task to be completed
    train_data = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train_data}

    def run(self):
        train_path = Path(self.requires()["train"].workdir).joinpath("train")
        background_audio = list(train_path.glob(f"{BACKGROUND_NOISE}/*.wav"))
        assert len(background_audio) > 0

        # Read all the background audio files and split into 1 second segments,
        # save all the segments into a folder called _silence_
        silence_dir = os.path.join(self.workdir, SILENCE)
        os.makedirs(silence_dir, exist_ok=True)

        print("Generating silence files from background sounds ...")
        for audio_path in tqdm(background_audio):
            audio, sr = sf.read(str(audio_path))

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


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
        }

    """
    @staticmethod
    def apply_label(relative_path):
        label = os.path.basename(os.path.dirname(relative_path))
        if label not in WORDS and label != SILENCE:
            label = UNKNOWN
        return label
    """

    def get_split_paths(self):

        # Test files
        test_path = Path(self.requires()["test"].workdir).joinpath("test")
        test_df = pd.DataFrame(test_path.glob("*/*.wav"), columns=["relpath"]).assign(
            split=lambda df: "test"
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
            split=lambda df: "valid"
        )

        # Train files
        # [really?? why is it called testing_list?]
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
            split=lambda df: "train"
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
            # [what does this mean??]
            # in the readme of google speech commands. we want to keep similar people
            # in the same group - test or train or val
            # [so do we do this or not?]
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

    def get_process_metadata(self) -> pd.DataFrame:
        process_metadata = self.get_split_paths()
        process_metadata = self.get_metadata_attrs(process_metadata)
        return process_metadata


def main(num_workers: int, sample_rates: List[int]):

    download_tasks = pipeline.get_download_and_extract_tasks(config)

    generate = GenerateTrainDataset(
        train_data=download_tasks["train"], data_config=config
    )
    configure_metadata = ExtractMetadata(
        train=generate,
        test=download_tasks["test"],
        outfile="process_metadata.csv",
        data_config=config,
    )

    final = pipeline.FinalizeCorpus(
        sample_rates=sample_rates, metadata=configure_metadata, data_config=config
    )

    pipeline.run(final, num_workers=num_workers)
