#!/usr/bin/env python3
"""
Profile GPU maximum memory usage.
"""

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

nvmlInit()

max_memory_used = None


def reset():
    global max_memory_used
    max_memory_used = None


def measure() -> float:
    global max_memory_used
    if torch.cuda.is_available():
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        if info.used > max_memory_used:
            max_memory_used = info.used
    return max_memory_used
