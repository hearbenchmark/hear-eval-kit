"""
A builder class that helps to construct luigi dataset pre-processing pipelines
"""
import os
from urllib.parse import urlparse
from typing import Any, Dict, List, Union
from types import MethodType, new_class
from functools import partial

import luigi

from heareval.tasks.config import DatasetConfig, get_config
import heareval.tasks.util.luigi as luigi_util


class DatasetBuilder:
    def __init__(self, task_config: Union[str, DatasetConfig]):
        if isinstance(task_config, str):
            self.config = get_config(task_config)
        elif isinstance(task_config, DatasetConfig):
            self.config = task_config
        else:
            raise TypeError(
                f"task_config must be a str or a DataConfig object, "
                f"received {type(task_config)}."
            )

        # Set the task name for all luigi WorkTasks
        luigi_util.WorkTask.task_name = self.config.task_name

    @staticmethod
    def add_requirements(requirements: Union[luigi.Task, List, Dict], namespace: Dict):
        """
        Given a namespace this adds a requires() class method that returns
        the requirments object passed in.

        Args:
            requirements: A requirements object to return in the new class method
            namespace: namespace to add the requires method to
        """
        namespace["requires"] = lambda self: requirements

    def build_task(
        self,
        base: Any,
        name: str = None,
        requirements: Union[luigi.Task, List, Dict] = None,
    ) -> Any:
        """
        Dynamically creates a luigi work task with the passed in requirements

        Args:
            base: Base class for this task
            name: Optional unique name for the new task class. Defaults to the
                same name as the base class. If you need a separate output folder
                for this task if there are multiple, or a uniquely named class
                then make sure to set this.
            requirements: Optional requirements that will be returned by the requires()
                method of the newly created class.

        Returns:
            A new class that derives from the base class

        """
        task_class = new_class(
            name=name,
            bases=(base,),
            exec_body=partial(self.add_requirements, requirements),
        )
        return task_class

    def download_and_extract_tasks(self) -> List[luigi_util.WorkTask]:
        """
        Builds the download and extract tasks from a dictionary of download urls
        """
        tasks = []

        # For each required download in the config, creates a new ExtractArchive
        # class with the correct Download requirement. The new ExtractArchive tasks
        # are returned as a list of tasks.
        for name, url in self.config.download_urls.items():
            filename = os.path.basename(urlparse(url).path)
            task = self.build_task(
                base=luigi_util.ExtractArchive,
                name=f"ExtractArchive{name.lower().title()}",
                requirements={
                    "download": luigi_util.DownloadCorpus(url=url, outfile=filename)
                },
            )
            tasks.append(task(infile=filename))

        return tasks
