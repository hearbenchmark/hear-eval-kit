#!/usr/bin/env python3
"""
Performs evaluation on embedding predictions.
"""

from pathlib import Path

import click
from tqdm import tqdm

from heareval.evaluation.task_evaluation import task_evaluation


@click.command()
@click.option(
    "--embeddings-dir",
    default="embeddings",
    help="Location of task embeddings to compute evaluation on.",
    type=click.Path(exists=True),
)
def runner(embeddings_dir: str = "embeddings") -> None:
    embeddings_dir_path = Path(embeddings_dir)
    if not embeddings_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing embeddings. "
            f"Ensure that directory named {embeddings_dir_path} exists."
        )

    embeddings = list(embeddings_dir_path.iterdir())

    for embedding in tqdm(embeddings):
        print(f"Evaluating model: {embedding.name}", flush=True)

        tasks = list(embedding.iterdir())
        for task_path in tasks:
            print(f"  - Evaluating task: {task_path.name}")
            task_evaluation(task_path)


if __name__ == "__main__":
    runner()
