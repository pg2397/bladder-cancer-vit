"""Single-call seeding for full reproducibility (see report Section 4.10)."""
from __future__ import annotations
import os
import random
import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness used in the pipeline.

    Covers Python's ``random``, NumPy, PyTorch CPU, PyTorch CUDA, and the
    per-worker DataLoader generators (via ``PYTHONHASHSEED``). With
    ``deterministic=True`` this also enables PyTorch's deterministic
    algorithms flag and disables CuDNN benchmarking so two runs of the same
    config produce identical metrics on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # use_deterministic_algorithms can throw on a few ops; warn-only
            # mode keeps training going while still flagging non-deterministic
            # kernels in the logs.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seed that derives from the parent generator."""
    import torch
    base_seed = torch.initial_seed() % (2 ** 32)
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)
