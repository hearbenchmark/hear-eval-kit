#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import click
import torch
from tqdm import tqdm

from heareval.predictions.task_predictions import task_predictions


@click.command()
@click.argument(
    "task_dirs",
    nargs=-1,
    required=True,
)
@click.option(
    "--grid-points",
    default=8,
    help="Number of grid points for randomized grid search "
    "model selection. (Default: 8)",
    type=click.INT,
)
@click.option(
    "--gpus",
    default=None if not torch.cuda.is_available() else "[0]",
    help='GPUs to use, as JSON string (default: "[0]" if any '
    "are available, none if not). "
    "See https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices",  # noqa
    type=str,
)
@click.option(
    "--in-memory",
    default=True,
    help="Load embeddings in memory, or memmap them from disk. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--deterministic",
    default=True,
    help="Deterministic or non-deterministic. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--grid",
    default="default",
    help='Grid to use: ["default", "fast", "faster"]',
    type=str,
)
def runner(
    task_dirs: Tuple[str],
    grid_points: int = 8,
    gpus: Any = None if not torch.cuda.is_available() else "[0]",
    in_memory: bool = True,
    deterministic: bool = True,
    grid: str = "default",
) -> None:
    if gpus is not None:
        gpus = json.loads(gpus)

    # random.shuffle(task_dirs)
    for task_dir in tqdm(task_dirs):
        task_path = Path(task_dir)
        print(f"Computing predictions for {task_path.name}")
        if not task_path.is_dir():
            raise ValueError(f"{task_path} should be a directory")

        train_embedding_dimensions = task_path.joinpath(
            "train.embedding-dimensions.json"
        )
        if not train_embedding_dimensions.exists():
            raise ValueError(f"{train_embedding_dimensions} does not exist")

        embedding_size = json.load(open(train_embedding_dimensions))[1]
        if (
            embedding_size
            != json.load(open(task_path.joinpath("valid.embedding-dimensions.json")))[1]
            or embedding_size
            != json.load(open(task_path.joinpath("test.embedding-dimensions.json")))[1]
        ):
            raise ValueError("Embedding dimension mismatch among JSON files")

        start = time.time()
        task_predictions(
            embedding_path=task_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            gpus=gpus,
            in_memory=in_memory,
            deterministic=deterministic,
            grid=grid,
        )
        sys.stdout.flush()
        print(
            f"DONE. took {time.time() - start} seconds to complete task_predictions"
            f"(embedding_path={task_path}, embedding_size={embedding_size}, "
            f"grid_points={grid_points}, gpus={gpus}, in_memory={in_memory}, "
            f"deterministic={deterministic}, grid={grid})"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    runner()
