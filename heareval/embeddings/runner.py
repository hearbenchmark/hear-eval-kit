#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""
from pathlib import Path

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
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Location of tasks to compute embeddings on",
    type=str,
)
def runner(module: str, model: str = None, tasks_dir: str = "tasks") -> None:

    # Check for directory containing the tasks
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    # Load the embedding model
    embedding = Embedding(module, model)

    tasks = list(tasks_dir.iterdir())
    for task_path in tqdm(tasks):
        print(f"Computing embeddings for {task_path.name}")
        task_embeddings(embedding, task_path)


if __name__ == "__main__":
    runner()
