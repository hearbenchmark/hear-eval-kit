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
from functools import partial
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
    "download_urls": {
        "train": "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip",  # noqa: E501
        "test": "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip",  # noqa: E501
    },
    # TODO: FIXME
    # Want different for train and test?
    "sample_duration": 120.0,
    "partitions": [
        {"name": "train", "max_files": 100},
        {"name": "test", "max_files": 100},
    ],
}


class ConfigureProcessMetaData(luigi_util.WorkTask):
    """
    Custom metadata pre-processing for the NSynth task. Creates a metadata csv
    file that will be used by downstream luigi tasks to curate the final dataset.
    """

    outfile = luigi.Parameter()
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
        }

    @staticmethod
    def get_rel_path(root: Path, item: pd.DataFrame) -> Path:
        # Creates the relative path to an audio file given the note_str
        audio_path = root.joinpath("audio")
        filename = f"{item}.wav"
        return audio_path.joinpath(filename)

    @staticmethod
    def slugify_file_name(filename: str) -> str:
        return f"{slugify(filename)}.wav"

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(split)
        split_path = split_path.joinpath(f"nsynth-{split}")

        metadata = pd.read_json(split_path.joinpath("examples.json"), orient="index")

        # Filter out pitches that are not within the range
        metadata = metadata[metadata["pitch"] >= config["pitch_range_min"]]
        metadata = metadata[metadata["pitch"] <= config["pitch_range_max"]]

        metadata = metadata.assign(label=lambda df: df["pitch"])
        metadata = metadata.assign(
            relpath=lambda df: df["note_str"].apply(
                partial(self.get_rel_path, split_path)
            )
        )
        metadata = metadata.assign(
            slug=lambda df: df["note_str"].apply(self.slugify_file_name)
        )
        metadata = metadata.assign(partition=lambda df: split)
        metadata = metadata.assign(
            filename_hash=lambda df: df["slug"].apply(luigi_util.filename_to_int_hash)
        )

        return metadata[luigi_util.PROCESSMETADATACOLS]

    def run(self):
        assert False

        # Get metadata for each of the data splits
        process_metadata = pd.concat(
            [self.get_split_metadata(split) for split in self.requires()]
        )

        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            columns=luigi_util.PROCESSMETADATACOLS,
            header=False,
            index=False,
        )

        self.mark_complete()


def main(num_workers: int, sample_rates: List[int]):

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(config)

    configure_metadata = ConfigureProcessMetaData(
        outfile="process_metadata.csv", data_config=config, **download_tasks
    )
    final = pipeline.FinalizeCorpus(
        sample_rates=sample_rates, metadata=configure_metadata, data_config=config
    )

    pipeline.run(final, num_workers=num_workers)
