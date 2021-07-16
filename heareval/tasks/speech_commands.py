#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os
import re
from functools import partial
from pathlib import Path

import luigi
import pandas as pd
import soundfile as sf
from tqdm import tqdm

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
    slugify_file_name,
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


class ConfigureTrainValMetaData(WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        return {
            "train": GenerateTrainDataset(),
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

        # Start creating a csv to store all the metadata for this data
        process_metadata = pd.DataFrame(train_paths, columns=["relpath"])
        process_metadata["partition"] = "train"

        # Add all the validation files
        val_metadata = pd.DataFrame(validation_paths, columns=["relpath"])
        val_metadata["partition"] = "validation"
        process_metadata = process_metadata.append(val_metadata)

        # Create a unique slug for each file. We include the folder with the class
        # name b/c the base filenames may not be unique.
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

        # Save the process metadata
        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


class ConfigureTestMetaData(WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        return {
            "test": ExtractArchiveTest(infile="test-corpus.tar.gz"),
        }

    @staticmethod
    def name_for_slug(path):
        # In the test set all the unknown sounds already have the class label
        # prepended to the name, so we don't want to add 'unknown' to them as well.
        label = os.path.basename(os.path.dirname(path))
        if label == UNKNOWN:
            path = os.path.basename(path)
        return path

    def run(self):
        # The testing set is all the files in the separate testing download
        test_path = Path(self.requires()["test"].workdir)
        test_paths = [str(p.relative_to(test_path)) for p in test_path.glob("*/*.wav")]

        process_metadata = pd.DataFrame(test_paths, columns=["relpath"])
        process_metadata["partition"] = "test"

        # Create a unique slug for each file. We include the folder with the class
        # name b/c the base filenames may not be unique.
        process_metadata["slug"] = (
            process_metadata["relpath"]
            .apply(self.name_for_slug)
            .apply(slugify_file_name)
        )
        process_metadata["relpath"] = process_metadata["relpath"].apply(
            partial(os.path.join, test_path)
        )

        # Hash the field id rather than the full path.
        process_metadata["filename_hash"] = (
            process_metadata["slug"]
            .apply(lambda relpath: re.sub(r"-nohash-.*$", "", relpath))
            .apply(filename_to_int_hash)
        )

        # Get label for the data from anywhere.
        # In this case it is the folder name
        process_metadata["label"] = process_metadata["relpath"].apply(
            ConfigureTrainValMetaData.apply_label
        )

        # Save the process metadata
        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


class CombineMetaData(WorkTask):
    """
    Because we dealt with the metadata for the train/val and testing datasets
    separately this combines them together into a single csv file.
    """

    outfile = luigi.Parameter()

    def requires(self):
        return {
            "train": ConfigureTrainValMetaData(outfile="train_corpus_metadata.csv"),
            "test": ConfigureTestMetaData(outfile="test_corpus_metadata.csv"),
        }

    def run(self):
        train = self.requires()["train"]
        train_meta = pd.read_csv(
            os.path.join(train.workdir, train.outfile),
            header=None,
            names=PROCESSMETADATACOLS,
        )

        test = self.requires()["test"]
        test_meta = pd.read_csv(
            os.path.join(test.workdir, test.outfile),
            header=None,
            names=PROCESSMETADATACOLS,
        )

        process_metadata = train_meta.append(test_meta)
        # Save the process metadata
        process_metadata.to_csv(
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
            "meta": CombineMetaData(outfile="process_metadata.csv"),
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
            "meta": CombineMetaData(outfile="process_metadata.csv"),
        }


class SplitTrainTestMetadata(SplitTrainTestMetadata):
    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(),
            "meta": CombineMetaData(outfile="process_metadata.csv"),
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
