"""One seed policy for the whole pipeline (UPGRADE_PLAN.md 0.6).

main.py, the experiment runner, and the noise simulator each carried their own
copy of this function. Divergence between them is a reproducibility bug, so
there is now exactly one.
"""
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed numpy, the stdlib RNG, and PennyLane."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import pennylane as qml
        qml.set_seed(seed)
    except Exception:
        # PennyLane exposes set_seed only in some builds; numpy seeding still applies.
        pass
