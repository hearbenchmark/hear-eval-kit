"""
Utility function to help
1. Change the luigi config and run tasks
2. Run the tasks in a different subprocess so that pytest can work
"""

import sys
from pathlib import Path
from importlib import import_module

from multiprocessing import Process
from mock import patch
from contextlib import ExitStack

from heareval.tasks.runner import run


def run_luigi_test_pipeline():
    """
    Runs the pipeline by picking arguments from the sys
    The main requirement of this function is to run luigi pipeline in a subprocess
    so that pytest can work by running multiple pipelines and comparing there results
    """
    temp_dir, task, dataset_fraction, suffix = sys.argv[1:]
    temp_dir = Path(temp_dir)

    # Get the config for the pipeline and change the config if the test requires to
    # change the config. This helps to run the pipeline by selectively mocking the
    # configuration
    task_module = import_module(f"heareval.tasks.{task}")
    task_config = task_module.config
    task_config["small"].update(
        {
            "dataset_fraction": float(dataset_fraction),
            "version": "{suffix}_{version}".format(
                suffix=suffix, version=task_config["small"]["version"]
            ),
        }
    )

    # Enter multiple contexts
    with ExitStack() as stack:
        # First context to mock the config with the parameters for the test
        stack.enter_context(patch(f"heareval.tasks.{task}.config", task_config))
        # Second to call the click command in the runner with the appropriate command
        # for the task
        stack.enter_context(
            patch(
                "sys.argv",
                (
                    f"heareval.tasks.runner {task} --small"
                    f" --luigi-dir {temp_dir.joinpath('luigi_dir')}"
                    f" --tasks-dir {temp_dir.joinpath('task_dir')}"
                ).split(),
            )
        )
        # Run the process
        p = Process(target=run())
        p.start()
        # Wait for the process to complete
        p.join()

    return temp_dir


if __name__ == "__main__":
    run_luigi_test_pipeline()
