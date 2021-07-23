#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import os
from pathlib import Path
from functools import partial
import logging
from typing import List

import luigi
import pandas as pd
from slugify import slugify

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")


config = {
    "task_name": "nsynth_pitch",
    "version": "v2.2.3",
    "download_urls": {
        "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa: E501
        "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa: E501
        "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa: E501
    },
    "sample_duration": 4.0,
    "partitions": [
        {"name": "train", "max_files": 100},
        {"name": "test", "max_files": 100},
        {"name": "valid", "max_files": 100},
    ],
    "pitch_range": [21, 108],
}


class ConfigureProcessMetaData(luigi_util.WorkTask):
    """
    Custom metadata pre-processing for the NSynth task. Creates a metadata csv
    file that will be used by downstream luigi tasks to curate the final dataset.
    """

    outfile = luigi.Parameter()
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    valid = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
            "valid": self.valid,
        }

    @staticmethod
    def get_rel_path(root: Path, item: pd.DataFrame) -> str:
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
        metadata = metadata[metadata["pitch"] >= config["pitch_range"][0]]
        metadata = metadata[metadata["pitch"] <= config["pitch_range"][1]]

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
