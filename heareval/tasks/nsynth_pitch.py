#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import os
from pathlib import Path
from functools import partial
import logging

import luigi
import pandas as pd
from slugify import slugify

from heareval.tasks.config import NSynthPitchConfig
from heareval.tasks.util.dataset_builder import DatasetBuilder
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")
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


def main():

    builder = DatasetBuilder(config)

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = builder.download_and_extract_tasks()
    configure_metadata = builder.build_task(
        ConfigureProcessMetaData,
        requirements=download_tasks,
        kwargs={"outfile": "process_metadata.csv"},
    )
    audio_tasks = builder.prepare_audio_from_metadata_task(configure_metadata)

    builder.run(audio_tasks)


if __name__ == "__main__":
    main()
