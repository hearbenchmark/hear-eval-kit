#!/usr/bin/env python3
"""
Pre-processing pipeline for DCASE 2016 task 2 task (sound event
detection).

The HEAR 2021 variation of DCASE 2016 Task 2 is that we ignore the
monophonic training data and use the dev data for train.
We also allow training data outside this task.
"""

import logging
import os
from pathlib import Path
from typing import List

import luigi
import pandas as pd
from slugify import slugify

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")


config = {
    "task_name": "dcase2016_task2",
    "version": "hear2021",
    # TODO: Use these somewhere
    "task_type": "event_labeling",
    "download_urls": {
        "train": "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip",  # noqa: E501
        "test": "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip",  # noqa: E501
    },
    # TODO: FIXME
    # Want different for train and test?
    "sample_duration": 120.0,
    "partitions": [
        {"name": "train", "max_files": 10},
        {"name": "test", "max_files": 10},
    ],
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train, "test": self.test}

    def split_to_path_str(self, split: str) -> str:
        """
        DCASE 2016 uses funny pathing, so we just hardcode the desired
        (paths)
        Note that for our training data, we only use DCASE 2016 dev data.
        Their training data is short monophonic events.
        """
        return {
            "train": "dcase2016_task2_train_dev/dcase2016_task2_dev/",
            "test": "dcase2016_task2_test_public/",
        }[split]

    def get_split_metadata(self, split: str) -> pd.DataFrame:
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
            sound_file = (
                str(annotation_file)
                .replace("annotation", "sound")
                .replace(".txt", ".wav")
            )
            assert os.path.exists(sound_file)
            metadata = metadata.assign(relpath=sound_file)

            metadata = metadata.assign(
                slug=lambda df: df.relpath.apply(self.slugify_file_name)
            )
            metadata = metadata.assign(partition=lambda df: split)
            metadata = metadata.assign(
                filename_hash=lambda df: df["slug"].apply(
                    luigi_util.filename_to_int_hash
                )
            )
            metadatas.append(metadata)

        return pd.concat(metadatas)[luigi_util.PROCESSMETADATACOLS]


def main(num_workers: int, sample_rates: List[int]):

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(config)

    configure_metadata = ExtractMetadata(
        outfile="process_metadata.csv", data_config=config, **download_tasks
    )
    final = pipeline.FinalizeCorpus(
        sample_rates=sample_rates, metadata=configure_metadata, data_config=config
    )

    pipeline.run(final, num_workers=num_workers)
