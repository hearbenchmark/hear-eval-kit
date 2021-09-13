#!/usr/bin/env python3
"""
Given commands in sys.stdin (one per line), distribute them across GPUs.
"""

import os
import sys
import time
from multiprocessing import Pool, current_process

import torch
import tqdm

gpus = torch.cuda.device_count()
assert gpus < 10, "Otherwise, fix the code getting the GPU from current_process().name"


def runcmd(cmd):
    gpu = int(current_process().name[-1]) - 1
    start = time.time()
    # Will this cause contention? Or each process has a different env?
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.system(cmd)
    end = time.time()
    print(f"\n\n\nDONE {end-start} sec: {cmd}\n\n\n")
    sys.stdout.flush()


if __name__ == "__main__":
    cmds = sys.stdin.readlines()
    print(cmds)
    assert gpus
    with Pool(gpus) as p:
        r = list(tqdm.tqdm(p.imap(runcmd, cmds), total=len(cmds)))
