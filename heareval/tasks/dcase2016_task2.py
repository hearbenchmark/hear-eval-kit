#!/usr/bin/env python3
"""
Pre-processing pipeline for DCASE 2016 task 2 task (sound event
detection).

The HEAR 2021 variation of DCASE 2016 Task 2 is that we ignore the
monophonic training data and use the dev data for train.
We also allow training data outside this task.
"""

import logging
from pathlib import Path
from typing import List

import luigi
import pandas as pd

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_utils

logger = logging.getLogger("luigi-interface")

# This percentage should not be change as this decides
# the data in the split and hence is not a part of the config
VALIDATION_PERCENTAGE = 0.3
TESTING_PERCENTAGE = 0

config = {
    "task_name": "dcase2016_task2",
    "version": "hear2021",
    "embedding_type": "event",
    "prediction_type": "multilabel",
    "download_urls": [
        {
            "name": "train",
            "url": "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip",  # noqa: E501
            "md5": "4e1b5e8887159193e8624dec801eb9e7",
        },
        {
            "name": "test",
            "url": "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip",  # noqa: E501
            "md5": "ac98768b39a08fc0c6c2ddd15a981dd7",
        },
    ],
    # TODO: FIXME
    # Want different for train and test?
    "sample_duration": 120.0,
    "splits": [
        {"name": "train", "max_files": 10},
        {"name": "test", "max_files": 10},
        {"name": "valid", "max_files": 10},
    ],
    "small": {
        "download_urls": [
            {
                "name": "train",
                "url": "https://github.com/turian/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_train_dev-small.zip",  # noqa: E501
                "md5": "3adca7e1860aedf7d1b0c06358f1b867",
            },
            {
                "name": "test",
                "url": "https://github.com/turian/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_test_public-small.zip",  # noqa: E501
                "md5": "73446e2156cb5f1d44a1de1e70e536a5",
            },
        ],
        "small_flag": True,
        "version": "hear2021-small",
        "splits": [
            {"name": "train", "max_files": 100},
            {"name": "test", "max_files": 100},
            {"name": "valid", "max_files": 100},
        ],
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train, "test": self.test}

    """
    DCASE 2016 uses funny pathing, so we just hardcode the desired
    (paths)
    Note that for our training data, we only use DCASE 2016 dev data.
    Their training data is short monophonic events.
    """
    split_to_path_str = {
        "train": "dcase2016_task2_train_dev/dcase2016_task2_dev/",
        "test": "dcase2016_task2_test_public/",
    }

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        # Since the valid is part of the train
        if split not in ["train", "test"]:
            return pd.DataFrame()
        logger.info(f"Preparing metadata for {split}")

        split_path = (
            Path(self.requires()[split].workdir)
            .joinpath(split)
            .joinpath(self.split_to_path_str[split])
        )

        metadatas = []
        for annotation_file in split_path.glob("annotation/*.txt"):
            metadata = pd.read_csv(
                annotation_file, sep="\t", header=None, names=["start", "end", "label"]
            )
            # Convert start and end times to milliseconds
            metadata["start"] *= 1000
            metadata["end"] *= 1000
            sound_file = (
                str(annotation_file)
                .replace("annotation", "sound")
                .replace(".txt", ".wav")
            )
            # Remove the assert statement as this file might not exist.
            # This is anyways ensured in the Base ExtractMetadata task.
            # assert os.path.exists(sound_file)
            metadata = metadata.assign(
                relpath=sound_file,
                slug=lambda df: df.relpath.apply(self.slugify_file_name),
                subsample_key=lambda df: self.get_subsample_key(df),
                # For train the split is decided by the which set function.
                # which takes in validation and testing percentage
                split=lambda df: split
                if split == "test"
                else df["subsample_key"].apply(
                    lambda filename_hash: luigi_utils.which_set(
                        filename_hash, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE
                    )
                ),
                split_key=lambda df: self.get_split_key(df),
                stratify_key=lambda df: self.get_stratify_key(df),
            )

            metadatas.append(metadata)

        return pd.concat(metadatas)


def main(
    num_workers: int,
    sample_rates: List[int],
    luigi_dir: str,
    tasks_dir: str,
    small: bool = False,
):
    if small:
        config.update(config["small"])
    config.update({"luigi_dir": luigi_dir})

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(config)

    configure_metadata = ExtractMetadata(
        outfile="process_metadata.csv", data_config=config, **download_tasks
    )
    final = pipeline.FinalizeCorpus(
        sample_rates=sample_rates,
        tasks_dir=tasks_dir,
        metadata=configure_metadata,
        data_config=config,
    )

    pipeline.run(final, num_workers=num_workers)
