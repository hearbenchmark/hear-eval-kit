#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from pathlib import Path

import luigi
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from slugify import slugify

import heareval.tasks.config.speech_commands as config
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
            partition=lambda df: "validation"
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
