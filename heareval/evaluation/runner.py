#!/usr/bin/env python3
"""
Performs evaluation on embedding predictions.
"""

import json
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
@click.option(
    "--embedding-version",
    default="all",
    help="Versioned embedding name in the embeddings dir to evaluate",
)
def runner(embeddings_dir: str = "embeddings", embedding_version: str = "all") -> None:
    embeddings_dir_path = Path(embeddings_dir)
    if not embeddings_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing embeddings. "
            f"Ensure that directory named {embeddings_dir_path} exists."
        )

    embeddings = (
        list(embeddings_dir_path.iterdir())
        if embedding_version == "all"
        else [embeddings_dir_path.joinpath(embedding_version)]
    )

    for embedding in tqdm(embeddings):
        print(f"Evaluating model: {embedding.name}", flush=True)

        # If it is not a directory it is the evaluation.json file from a previous run
        # which needs to be skipped
        tasks = [path for path in embedding.iterdir() if path.is_dir()]
        embedding_results = {}
        for task_path in tqdm(tasks):
            embedding_results[task_path.name] = task_evaluation(task_path)

        # Dump evaluation json in the embedding_task folder
        json.dump(
            embedding_results,
            embedding.joinpath("evaluation_results.json").open("w"),
            indent=4,
        )


if __name__ == "__main__":
    runner()
