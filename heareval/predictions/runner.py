#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.
"""
from pathlib import Path

import click
from tqdm import tqdm

from heareval.predictions.task_predictions import task_predictions


@click.command()
@click.argument("module", type=str)
@click.option(
    "--embeddings-dir",
    default="embeddings",
    help="Location of tasks to compute embeddings on",
    type=click.Path(exists=True),
)
def runner(module: str, embeddings_dir: str = None) -> None:
    embeddings_dir = embeddings_dir.joinpath(module)
    if not embeddings_dir.is_dir():
        raise ValueError(
            "Cannot locate directory containing embeddings. "
            f"Ensure that directory named {embeddings_dir} exists."
        )

    tasks = list(embeddings_dir.iterdir())
    for task_path in tqdm(tasks):
        print(f"Computing predictions for {task_path.name}")
        task_predictions(task_path)


if __name__ == "__main__":
    runner()
