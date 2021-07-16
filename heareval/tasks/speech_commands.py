#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from functools import partial
from glob import glob
from pathlib import Path
import logging

import heareval.tasks.config.speech_commands as config
import luigi
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

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

WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
BACKGROUND_NOISE = "_background_noise_"
UNKNOWN = "_unknown_"
SILENCE = "_silence_"


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


class GenerateTrainDataset(WorkTask):
    """
    Silence / background samples in the train / validation silents need to be
    created by slicing up longer background samples into 1sec slices
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


class ConfigureProcessMetaData(WorkTask):
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

    def run(self):

        train_path = Path(self.requires()["train"].workdir)

        # List of all relative paths to the testing and validation files
        with open(os.path.join(train_path, "testing_list.txt"), "r") as fp:
            train_test_paths = fp.read().strip().splitlines()

        with open(os.path.join(train_path, "validation_list.txt"), "r") as fp:
            validation_paths = fp.read().strip().splitlines()

        audio_paths = [
            str(p.relative_to(train_path)) for p in train_path.glob("[!_]*/*.wav")
        ]

        # The training set is files that are not in either the train-test or validation.
        train_paths = set(audio_paths) - set(train_test_paths) - set(validation_paths)
        train_paths = list(train_paths)

        # Manually add in the silent samples
        all_silence = [
            str(p.relative_to(train_path)) for p in train_path.glob(f"{SILENCE}/*.wav")
        ]
        val_silence = [
            str(p.relative_to(train_path))
            for p in train_path.glob(f"{SILENCE}/running_tap*.wav")
        ]
        train_silence = list(set(all_silence) - set(val_silence))

        train_paths.extend(train_silence)
        validation_paths.extend(val_silence)

        # The testing set is all the files in the separate testing download
        # test_path = Path(self.requires()["test"].workdir)
        # test_paths = [p.relative_to(test_path) for p in test_path.glob("*/*.wav")]

        process_metadata = pd.DataFrame(train_paths, columns=["relpath"])
        process_metadata["partition"] = "train"

        # Add all the validation files
        val_metadata = pd.DataFrame(validation_paths, columns=["relpath"])
        val_metadata["partition"] = "validation"
        process_metadata = process_metadata.append(val_metadata)

        process_metadata["slug"] = process_metadata["relpath"].apply(slugify_file_name)
        process_metadata["relpath"] = process_metadata["relpath"].apply(
            partial(os.path.join, train_path)
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

        # Get label for the data from anywhere.
        # In this case it is the folder name
        process_metadata["label"] = process_metadata["relpath"].apply(self.apply_label)

        # Get unique slug for each file
        # In this case the base name is not excluded since same person have made dataset
        # for different category
        # This is something which is data dependent and should be handled here
        process_metadata["slug"] = (
            process_metadata["relpath"]
            # Dont exclude the base name. Apply sluggify on all
            .apply(partial(os.path.relpath, start=train_path)).apply(slugify_file_name)
        )

        # Save the process metadata
        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


class SubsampleCorpus(SubsampleCorpus):
    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {"meta": ConfigureProcessMetaData(outfile="train_corpus_metadata.csv")}


class MonoWavTrimCorpus(MonoWavTrimCorpus):
    def requires(self):
        return {"corpus": SubsampleCorpus(max_files=config.MAX_FILES_PER_CORPUS)}


class SplitTrainTestCorpus(SplitTrainTestCorpus):
    def requires(self):
        # The metadata helps in provide the partition type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(duration=config.SAMPLE_LENGTH_SECONDS),
            "meta": ConfigureProcessMetaData(outfile="train_corpus_metadata.csv"),
        }


class SplitTrainTestMetadata(SplitTrainTestMetadata):
    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(),
            "meta": ConfigureProcessMetaData(outfile="train_corpus_metadata.csv"),
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
