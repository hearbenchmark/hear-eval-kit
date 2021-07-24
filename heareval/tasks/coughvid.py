#!/usr/bin/env python3
"""
Pre-processing pipeline for Coughvid Dataset
"""

import logging
import os
from pathlib import Path
from typing import List
from functools import partial

import luigi
import pandas as pd
from slugify import slugify

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")

VALID_EXTENSIONS = (".webm", ".ogg")
# These percentages should be fixed and should not change for a dataset.
# The size of the set can be altered by the split size to take.
# This does the initial split of the metadata which is not done in the orignal dataset
VALIDATION_PERCENTAGE = 30
TESTING_PERCENTAGE = 10

config = {
    "task_name": "coughvid",
    "version": "hear2021",
    "task_type": "scene_labeling",
    "download_urls": {
        "all_data": "https://zenodo.org/record/4498364/files/public_dataset.zip"
    },
    "sample_duration": 2.0,
    "splits": [
        {"name": "train", "max_files": 10},
        {"name": "test", "max_files": 10},
        {"name": "valid", "max_files": 10},
    ],
}


class ExtractMetadata(pipeline.ExtractMetadata):
    all_data = luigi.TaskParameter()

    def requires(self):
        return {"all_data": self.all_data}

    @staticmethod
    def slugify_file_name(relative_path: str) -> str:
        # Override the original slugify_file_name as in this case the extensions
        # might be different
        name, ext = os.path.splitext(os.path.basename(relative_path))
        return f"{slugify(str(name))}"

    def get_process_metadata(self) -> pd.DataFrame:
        logger.info(f"Preparing metadata")

        all_data_path = Path(self.requires()["all_data"].workdir).joinpath(
            os.path.join("all_data", "public_dataset")
        )
        # Prepare the metadata by reading all the files in the downloaded folder
        print("##########")
        print(list(all_data_path.glob("*"))[:10])
        print("=========")
        print(
            list(
                filter(
                    lambda p: p.suffix in VALID_EXTENSIONS,
                    list(all_data_path.glob("*")),
                )
            )[:10]
        )
        metadata = pd.DataFrame(
            # Read all the files with valid extension in the corpus folder
            filter(
                lambda p: p.suffix in VALID_EXTENSIONS, list(all_data_path.glob("*"))
            ),
            columns=["relpath"],
        ).assign(
            # Get uuid to join with the basemetadata and get the label
            uuid=lambda df: df["relpath"].apply(
                lambda p: os.path.splitext(os.path.basename(p))[0]
            ),
        )

        # Read the base metadata to get the label and the valid examples
        base_metadata = (
            pd.read_csv(os.path.join(all_data_path, "metadata_compiled.csv"))
            # Filter the data points with null status
            .loc[lambda df: df["status"].notnull()]
            # Rename the status as the label
            .rename(columns={"status": "label"}).filter(items=["uuid", "label"])
        )

        # Define the which set which will decide the split for each dataset
        which_set = partial(
            luigi_util.which_set,
            validation_percentage=VALIDATION_PERCENTAGE,
            testing_percentage=TESTING_PERCENTAGE,
        )

        # Do an inner join
        metadata = metadata.merge(base_metadata, on=["uuid"], how="inner").assign(
            slug=lambda df: df["relpath"].apply(self.slugify_file_name),
            filename_hash=lambda df: df["slug"].apply(luigi_util.filename_to_int_hash),
            split=lambda df: df["filename_hash"].apply(which_set),
        )
        print(metadata.head(5))

        return metadata[["relpath", "slug", "filename_hash", "split", "label"]]


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


if __name__ == "__main__":
    main(2, [16000, 22050])
