#!/usr/bin/env python3
"""
Pre-processing pipeline for Google Speech Commands
"""
# from heareval.tasks.coughvid import DownloadCorpus, ExtractCorpus
import os
import subprocess

import luigi
import numpy as np

import heareval.tasks.config.speech_commands as config
import heareval.tasks.util.audio as audio_util
import heareval.tasks.util.luigi as luigi_util
from heareval.tasks.util.luigi import DownloadCorpus, WorkTask, ExtractArchive


# Set the task name for all WorkTasks
WorkTask.task_name = config.TASKNAME


def main():
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    luigi_util.ensure_dir("_workdir")
    
    download_corpus = DownloadCorpus(url = config.DOWNLOAD_URL, outfile = 'corpus.tar.gz')
    extract_download_corpus = ExtractArchive(infile = 'corpus.tar.gz', prev_task = download_corpus)
    
    #Make the final task that needs to run. 
    #This will be the last task and luigi will work bottom up.
    final_task = extract_download_corpus
    
    #Run the final task
    luigi.build(
        [final_task], workers=config.NUM_WORKERS, local_scheduler=True, log_level="INFO"
    )

if __name__ == "__main__":
    main()
