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

from heareval.tasks.config.dataset_config import (
    PartitionedDatasetConfig,
    PartitionConfig,
)
from heareval.tasks.util.dataset_builder import DatasetBuilder
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")


# Dataset configuration
class NSynthPitchConfig(PartitionedDatasetConfig):
    def __init__(self):
        super().__init__(
            task_name="nsynth-pitch",
            version="v2.2.3",
            download_urls={
                "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa: E501
                "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa: E501
                "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa: E501
            },
            # All samples will be trimmed / padded to this length
            sample_duration=4.0,
            # Pre-defined partitions in the dataset. Number of files in each split is
            # train: 85,111; valid: 10,102; test: 4890.
            # To subsample a partition, set the max_files to an integer.
            partitions=[
                PartitionConfig(name="train", max_files=1000),
                PartitionConfig(name="valid", max_files=100),
                PartitionConfig(name="test", max_files=100),
            ],
        )


config = NSynthPitchConfig()


class ConfigureProcessMetaData(luigi_util.WorkTask):
    """
    This config is data dependent and has to be set for each data
    """

    outfile = luigi.Parameter()

    def requires(self):
        raise NotImplementedError

    @staticmethod
    def get_rel_path(root: Path, item: pd.DataFrame) -> str:
        # Creates the audio relative path for a dataframe item
        audio_path = root.joinpath("audio")
        filename = f"{item}.wav"
        return audio_path.joinpath(filename)

    @staticmethod
    def slugify_file_name(filename: str) -> str:
        return f"{slugify(filename)}.wav"

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(f"nsynth-{split}")

        metadata = pd.read_json(split_path.joinpath("examples.json"), orient="index")

        # Filter out pitches that are not within the range of a standard piano
        metadata = metadata[metadata["pitch"] >= 21]
        metadata = metadata[metadata["pitch"] <= 108]

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

    builder = DatasetBuilder(config)

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = builder.download_and_extract_tasks()
    configure_metadata = builder.build_task(
        ConfigureProcessMetaData,
        requirements=download_tasks,
        kwargs={"outfile": "process_metadata.csv"},
    )
    audio_tasks = builder.prepare_audio_from_metadata_task(
        configure_metadata, sample_rates
    )

    builder.run(audio_tasks, num_workers=num_workers)


if __name__ == "__main__":
    main()
