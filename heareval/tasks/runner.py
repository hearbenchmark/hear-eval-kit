#!/usr/bin/env python3
"""
Runs a luigi pipeline to build a dataset
"""

import logging
import multiprocessing
from typing import Optional

import click

import heareval.tasks.nsynth_pitch as nsynth_pitch
import heareval.tasks.speech_commands as speech_commands

logger = logging.getLogger("luigi-interface")

tasks = {"speech_commands": speech_commands, "nsynth_pitch": nsynth_pitch}


@click.command()
@click.argument("task")
@click.option(
    "--num-workers",
    default=None,
    help="Number of CPU workers to use when running. "
    "If not provided all CPUs are used.",
    type=int,
)
@click.option(
    "--sample-rate",
    default=None,
    help="Perform resampling only to this sample rate. "
    "By default we resample to 16000, 22050, 44100, 48000.",
    type=int,
)
def run(
    task: str, num_workers: Optional[int] = None, sample_rate: Optional[int] = None
):

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        logger.info(f"Using {num_workers} workers")

    if sample_rate is None:
        sample_rates = [16000, 22050, 44100, 48000]
    else:
        sample_rates = [sample_rate]

    tasks[task].main(num_workers=num_workers, sample_rates=sample_rates)


if __name__ == "__main__":
    run()
