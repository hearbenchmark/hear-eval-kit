#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import logging
from functools import partial
from pathlib import Path
from typing import List

import luigi
import pandas as pd

import heareval.tasks.pipeline as pipeline

logger = logging.getLogger("luigi-interface")


config = {
    "task_name": "nsynth_pitch",
    "version": "v2.2.3",
    "task_type": "scene_labeling",
    "download_urls": {
        "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa: E501
        "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa: E501
        "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa: E501
    },
    "sample_duration": 4.0,
    "splits": [
        {"name": "train", "max_files": 100},
        {"name": "test", "max_files": 100},
        {"name": "valid", "max_files": 100},
    ],
    "pitch_range_min": 21,
    "pitch_range_max": 108,
}


class ExtractMetadata(pipeline.ExtractMetadata):
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
    def get_rel_path(root: Path, item: pd.DataFrame) -> Path:
        # Creates the relative path to an audio file given the note_str
        audio_path = root.joinpath("audio")
        filename = f"{item}.wav"
        return audio_path.joinpath(filename)

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(split)
        split_path = split_path.joinpath(f"nsynth-{split}")

        metadata = pd.read_json(split_path.joinpath("examples.json"), orient="index")

        metadata = (
            # Filter out pitches that are not within the range
            metadata.loc[
                metadata["pitch"].between(
                    config["pitch_range_min"], config["pitch_range_max"]
                )
                # Assign metadata columns
            ].assign(
                label=lambda df: df["pitch"],
                relpath=lambda df: df["note_str"].apply(
                    partial(self.get_rel_path, split_path)
                ),
                slug=lambda df: df["note_str"].apply(self.slugify_file_name),
                split=lambda df: split,
                subsample_key=lambda df: df["slug"].apply(self.get_subsample_key),
            )
        )

        return metadata[["relpath", "slug", "subsample_key", "split", "label"]]


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
