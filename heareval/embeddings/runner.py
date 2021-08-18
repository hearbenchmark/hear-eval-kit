#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from heareval.embeddings.task_embeddings import Embedding, task_embeddings


@click.command()
@click.argument("module", type=str)
@click.option(
    "--model",
    default=None,
    help="Location of model weights file",
    type=click.Path(exists=True),
)
@click.option("--version", default="", help="Version of module and weights")
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Location of tasks to compute embeddings on",
    type=str,
)
@click.option(
    "--task-version",
    default="all",
    help="Versioned task name in the tasks dir to compute embedding on",
)
@click.option(
    "--embeddings-dir", default="embeddings", help="Location to save task embeddings"
)
def runner(
    module: str,
    version: str = "",
    model: str = None,
    tasks_dir: str = "tasks",
    task_version: Optional[str] = "all",
    embeddings_dir: str = "embeddings",
) -> None:

    # Check for directory containing the tasks
    tasks_dir_path = Path(tasks_dir)
    embeddings_dir_path = Path(embeddings_dir)
    if not tasks_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir_path} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    # Load the embedding model
    embedding = Embedding(module, model, str(version))

    tasks = (
        list(tasks_dir_path.iterdir())
        if task_version == "all"
        else [tasks_dir_path.joinpath(task_version)]
    )
    for task_path in tqdm(tasks):
        print(f"Computing embeddings for {task_path.name}")
        task_embeddings(embedding, task_path, embeddings_dir_path)


if __name__ == "__main__":
    runner()
