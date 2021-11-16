#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

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
    "--outdir",
    default=None,
    help="Output directory to save prediction results. (Defaults to "
         "the same directory as the input embeddings)",
    type=str,
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
    outdir: Optional[str] = None,
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

        # Get the output directory name
        if outdir is None:
            # Default is the task_path -- dir with embeddings
            outdir = task_path
        else:
            # A separate output directory was passed in.
            # Create a subdirectory within that with the same name as
            # the task to save the results in.
            outdir = Path(outdir).joinpath(task_path.name)
            outdir.mkdir(parents=True, exist_ok=True)

        # Get embedding sizes for all splits/folds
        metadata = json.load(task_path.joinpath("task_metadata.json").open())
        embedding_sizes = []
        for split in metadata["splits"]:
            split_path = task_path.joinpath(f"{split}.embedding-dimensions.json")
            embedding_sizes.append(json.load(split_path.open())[1])

        # Ensure all embedding sizes are the same across splits/folds
        embedding_size = embedding_sizes[0]
        if len(set(embedding_sizes)) != 1:
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
            outdir=outdir,
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
