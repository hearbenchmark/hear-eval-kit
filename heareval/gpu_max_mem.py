#!/usr/bin/env python3
"""
Profile GPU maximum memory usage.
"""

from typing import Optional

import torch

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print("WARNING: gpu_max_mem measures the *first* GPU, but you have several.")

    from pynvml import (
        NVMLError,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
    )

    nvmlInit()

    max_memory_used: Optional[float] = None

    def reset():
        global max_memory_used
        max_memory_used = None

    def measure() -> Optional[float]:
        global max_memory_used
        if torch.cuda.is_available():
            try:
                h = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(h)
                # Convert to GB
                memory_used: float = info.used / 1024 / 1024 / 1024
                if max_memory_used is None or memory_used > max_memory_used:
                    max_memory_used = memory_used
            except NVMLError:
                # Happens on Ubuntu 20.04 running on WSL2.
                pass
        return max_memory_used


else:

    def reset():
        pass

    def measure() -> Optional[float]:
        return None
