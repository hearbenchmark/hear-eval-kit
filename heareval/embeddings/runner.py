#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import os
from dataclasses import dataclass
from typing import Any, List, Union

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
from hydra.core.config_store import ConfigStore


@dataclass
class EmbeddingConfig:
    module: Any


cs = ConfigStore.instance()
cs.store(name="config", node=EmbeddingConfig)


@hydra.main(config_path=None, config_name="config")
def runner(cfg) -> None:

    print(get_original_cwd())
    if hasattr(cfg, "config"):
        override_path = hydra.utils.to_absolute_path(cfg.config)
        if os.path.isfile(override_path):
            override_conf = OmegaConf.load(override_path)
            cfg = OmegaConf.merge(cfg, override_conf)

    print(OmegaConf.to_yaml(cfg))
    print(cfg.module)


if __name__ == "__main__":
    runner()
