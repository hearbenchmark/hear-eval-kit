#!/usr/bin/env python3
"""
Runs a sampler to sample the downloaded dataset.

Uses the same download and extract tasks to make sure
the same downloaded files can be used for sampling
Also uses the configs defined in the task files for making
it simple to scale across multiple dataset
"""

import logging
import multiprocessing
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
import luigi
import pandas as pd
from tqdm import tqdm

import heareval.tasks.pipeline as pipeline
import heareval.tasks.util.luigi as luigi_util
from heareval.tasks import dcase2016_task2, nsynth_pitch, speech_commands

# Currently the sampler is only allowed to run for open tasks
# The secret tasks module will not be available for participants
try:
    import secret_tasks as secret_tasks
except ModuleNotFoundError:
    # For participants the secret_tasks_module will be None
    secret_tasks = None

logger = logging.getLogger("luigi-interface")

METADATAFORMATS = [".csv", ".json", ".txt"]
AUDIOFORMATS = [".mp3", ".wav", ".ogg"]


configs = {
    "nsynth_pitch": {
        "task_config": nsynth_pitch.config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    "speech_commands": {
        "task_config": speech_commands.config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    "dcase2016_task2": {
        "task_config": dcase2016_task2.config,
        "audio_sample_size": 4,
        # Put two files from the dev and train split so that those splits are
        # made
        # dev_1_ebr_6_nec_2_poly_0.wav -> 1 train file in valid split
        # dev_1_ebr_6_nec_3_poly_0.wav -> 1 valid file in valid split
        "necessary_keys": [
            "dev_1_ebr_6_nec_2_poly_0.wav",
            "dev_1_ebr_6_nec_3_poly_0.wav",
        ],
    },
    # Add the sampler config for the secrets task if the secret task config was found.
    # Not available for participants
    **(getattr(secret_tasks, "sampler_config") if secret_tasks else {}),
}


class RandomSampleOriginalDataset(luigi_util.WorkTask):
    necessary_keys = luigi.ListParameter()
    audio_sample_size = luigi.IntParameter()

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
                lambda file: any(key in str(file) for key in self.necessary_keys),
                all_files,
            )
        )
        audio_files_to_sample = pd.DataFrame(
            list(
                # Filter all the audio files which are not in the necessary list.
                # Out of these audios audio_sample_size number of samples will be
                # selected
                filter(
                    lambda file: file.suffix.lower() in map(str.lower, AUDIOFORMATS),
                    [file for file in all_files if file not in necessary_files],
                )
            ),
            columns=["audio_path"],
        )

        sampled_audio_files = (
            audio_files_to_sample.assign(
                # The subfolder name is set as the stratify key. This ensures at least
                # one audio is selected from each subfolder in the original dataset
                stratify_key=lambda df: df.audio_path.apply(lambda path: path.parent),
                # The split key is the hash of the path. This ensures the sampling is
                # deterministic
                subsample_key=lambda df: df.audio_path.apply(
                    lambda path: luigi_util.filename_to_int_hash(str(path))
                ),
                # Split key is same as the subsample key in this case
                split_key=lambda df: df.subsample_key,
            )
            # The above metadata is passed in the subsample metadata and
            # audio_sample_size number of files are selected
            .pipe(
                luigi_util.subsample_metadata, self.audio_sample_size
            ).audio_path.to_list()
        )

        return metadata_files + necessary_files + sampled_audio_files

    def run(self):
        for url_obj in self.data_config["small"]["download_urls"]:
            # Sample a small subset to copy from all the files
            url_name = Path(urlparse(url_obj["url"]).path).stem
            split = url_obj["name"]
            copy_from = self.requires()[split].workdir.joinpath(split)
            all_files = [file.relative_to(copy_from) for file in copy_from.rglob("*")]
            copy_files = self.sample(all_files)

            # Copy and make a zip
            copy_to = self.workdir.joinpath(url_name)
            if copy_to.exists():
                shutil.rmtree(copy_to)
            for file in tqdm(copy_files):
                self.safecopy(src=copy_from.joinpath(file), dst=copy_to.joinpath(file))
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
