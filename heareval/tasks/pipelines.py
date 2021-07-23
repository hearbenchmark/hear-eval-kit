"""
Generic pipelines for datasets
"""

import os
from urllib.parse import urlparse

from heareval.tasks.dataset_config import DatasetConfig
import heareval.tasks.util.luigi as luigi_util


def get_download_and_extract_tasks(config: DatasetConfig):

    tasks = {}
    for name, url in config["download_urls"].items():
        filename = os.path.basename(urlparse(url).path)
        task = luigi_util.ExtractArchive(
            download=luigi_util.DownloadCorpus(
                url=url, outfile=filename, data_config=config
            ),
            infile=filename,
            outdir=name,
            data_config=config,
        )
        tasks[name] = task

    return tasks
