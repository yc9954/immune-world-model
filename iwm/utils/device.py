"""Runtime device selection + thread tuning for Apple Silicon hosts.

Picks `mps` when available (Apple GPU), `cuda` on Nvidia, `cpu` otherwise.
Also raises `torch.set_num_threads` to the physical core count so linear
algebra kernels (used heavily by the synth generator, Ridge baseline, and
MotifBilinear Adam loop) stop idling at the torch default of 4.
"""

from __future__ import annotations

import os

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """`prefer` ∈ {'auto', 'mps', 'cuda', 'cpu'}."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lightning_accelerator(prefer: str = "auto") -> str:
    """Map device preference to Lightning's accelerator string."""
    d = pick_device(prefer)
    if d.type == "cuda":
        return "gpu"
    if d.type == "mps":
        return "mps"
    return "cpu"


def tune_threads(target: int | None = None) -> int:
    """Raise torch's intra-op thread count to physical cores on macOS.

    Returns the thread count actually applied.
    """
    if target is None:
        target = os.cpu_count() or 1
    torch.set_num_threads(int(target))
    # interop threads are already max-ed by torch on most builds; leave alone.
    return int(target)
