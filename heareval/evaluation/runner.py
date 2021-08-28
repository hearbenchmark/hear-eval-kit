#!/usr/bin/env python3
"""
Performs evaluation on embedding predictions.
"""

import json
import os
from pathlib import Path

import click
from tqdm import tqdm

from heareval.evaluation.task_evaluation import task_evaluation


@click.command()
@click.argument("module", type=str, default="all")
@click.option(
    "--task",
    default="all",
    help="Task to run. (Default: all)",
    type=str,
)
@click.option(
    "--embeddings-dir",
    default="embeddings",
    help="Location of task embeddings to compute evaluation on.",
    type=click.Path(exists=True),
)
def runner(
    module: str = "all", task: str = "all", embeddings_dir: str = "embeddings"
) -> None:
    embeddings_dir_path = Path(embeddings_dir)
    if not embeddings_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing embeddings. "
            f"Ensure that directory named {embeddings_dir_path} exists."
        )

    if module == "all":
        embeddings = list(embeddings_dir_path.iterdir())
    else:
        embeddings = [embeddings_dir_path.joinpath(module)]
        assert os.path.exists(embeddings[0]), f"{embeddings[0]} does not exist"

    results = {}
    for embedding in tqdm(embeddings):
        print(f"Evaluating model: {embedding.name}", flush=True)

        embedding_results = {}
        if task == "all":
            tasks = list(embedding.iterdir())
        else:
            tasks = [embedding.joinpath(task)]
            assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
        for task_path in tqdm(tasks):
            print(f"  - Evaluating task: {task_path.name}")
            embedding_results[task_path.name] = task_evaluation(task_path)

        results[embedding.name] = embedding_results

    # Save all the results to json
    json.dump(results, Path("evaluation_results.json").open("w"), indent=4)


if __name__ == "__main__":
    runner()
