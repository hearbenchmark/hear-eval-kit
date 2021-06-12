#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
import os

import luigi
import numpy as np

import heareval.tasks.config.speech_commands as config
import heareval.tasks.util.audio as audio_util
import heareval.tasks.util.luigi as luigi_util
from heareval.tasks.util.luigi import WorkTask


# Set the task name for all WorkTasks
WorkTask.task_name = config.TASKNAME


def main():
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    luigi_util.ensure_dir("_workdir")

    download = luigi_util.DownloadCorpus(
        url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        outfile="corpus.tar.gz",
    )

    extract = luigi_util.ExtractCorpus(infile="corpus.tar.gz")

    # Requires is a class method that returns a task / list of tasks that this
    # tasks depends upon
    extract.requires = lambda: download

    luigi.build(
        [extract], workers=config.NUM_WORKERS, local_scheduler=True, log_level="INFO"
    )


if __name__ == "__main__":
    main()
