#!/usr/bin/env python3
"""
Runs a sampler to sample the downloaded dataset.

Uses the same download and extract tasks to make sure
the same downloaded files can be used for sampling
Also uses the configs defined in the task files for making
it simple to scale across multiple dataset
"""

from pathlib import Path
from typing import Optional
import multiprocessing
import logging

import luigi
from tqdm import tqdm
from urllib.parse import urlparse
import shutil
import click
import pandas as pd

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util
from heareval.tasks.speech_commands import config as speech_command_config
from heareval.tasks.dcase2016_task2 import config as dcase2016_task2_config
from heareval.tasks.nsynth_pitch import config as nsynth_pitch_config

logger = logging.getLogger("luigi-interface")

METADATAFORMATS = [".csv", ".json", ".txt"]
AUDIOFORMATS = [".mp3", ".wav", ".ogg"]


configs = {
    "nsynth_pitch": {
        "task_config": nsynth_pitch_config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    "speech_commands": {
        "task_config": speech_command_config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    "dcase2016_task2": {
        "task_config": dcase2016_task2_config,
        "audio_sample_size": 10,
        # Add any keys in the file format that we compulsorily need to copy
        "necessary_keys": [],
    },
}


class RandomSampleOriginalDataset(luigi_util.WorkTask):
    necessary_keys = luigi.ListParameter()
    audio_sample_size = luigi.Parameter()

    def requires(self):
        return pipeline.get_download_and_extract_tasks(self.data_config)

    @staticmethod
    def safecopy(dst, src):
        # Make sure the parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def sample(self, all_files):
        # All metadata files will be copied without any sampling
        metadata_files = list(
            filter(lambda file: file.suffix in METADATAFORMATS, all_files)
        )
        # If the file name has a necessary key
        necessary_files = list(
            filter(
                lambda file: any(key in file for key in self.necessary_keys),
                all_files,
            )
        )
        # Audio files (leaving out the necessary files will be sampled)
        # Also stratify on the basis of base folder to make sure a better
        # snapshot is captured. For example in dcase the train and dev
        # are in same folder and the dev size is too small. Randomly
        # sampling without stratifying makes no dev folder which disrupts
        # the original folder structure. Using the subsample metadata functions
        # from luiti_utils solves this issue.
        metadata = pd.DataFrame(
            list(
                filter(
                    lambda file: file.suffix.lower() in map(str.lower, AUDIOFORMATS),
                    [file for file in all_files if file not in necessary_files],
                )
            ),
            columns=["audio_path"],
        )
        metadata = metadata.assign(
            stratify_key=lambda df: df.audio_path.apply(lambda path: path.parent),
            split_key=lambda df: df.audio_path.apply(
                lambda path: luigi_util.filename_to_int_hash(str(path))
            ),
            subsample_key=lambda df: df.split_key,
        )
        audio_files = luigi_util.subsample_metadata(
            metadata, self.audio_sample_size
        ).audio_path.to_list()
        
        return metadata_files + necessary_files + audio_files

    def run(self):
        for download_extract_task in self.data_config["download_urls"]:
            zip_download_name = Path(
                urlparse(download_extract_task["url"]).path
            ).name.split(".")[0]
            zip_extract_name = download_extract_task["name"]
            copy_to = self.workdir.joinpath(zip_download_name)
            # Remove the sampled folder if it already exists
            if copy_to.exists():
                shutil.rmtree(copy_to)
            copy_from = self.requires()[zip_extract_name].workdir.joinpath(
                zip_extract_name
            )
            copy_files = self.sample(list(copy_from.rglob("*")))
            for file in tqdm(copy_files):
                self.safecopy(
                    src=file, dst=copy_to.joinpath(file.relative_to(copy_from))
                )
            shutil.make_archive(copy_to, "zip", copy_to)


@click.command()
@click.argument("task")
@click.option(
    "--num-workers",
    default=None,
    help="Number of CPU workers to use when running. "
    "If not provided all CPUs are used.",
    type=int,
)
def main(task: str, num_workers: Optional[int] = None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} workers")
    config = configs[task]
    sampler = RandomSampleOriginalDataset(
        data_config=config["task_config"],
        audio_sample_size=config["audio_sample_size"],
        necessary_keys=config["necessary_keys"],
    )
    pipeline.run(sampler, num_workers=num_workers)


if __name__ == "__main__":
    main()
