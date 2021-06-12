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


class DownloadCorpus(WorkTask):
    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # TODO: Change the working dir
        luigi_util.download_file(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
            os.path.join(self.workdir, "corpus.zip"),
        )
        with self.output().open("w") as _:
            pass

    @property
    def stage_number(self) -> int:
        return 0


def main():
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    luigi_util.ensure_dir("_workdir")
    luigi.build([DownloadCorpus()], workers=config.NUM_WORKERS, local_scheduler=True)


if __name__ == "__main__":
    main()
