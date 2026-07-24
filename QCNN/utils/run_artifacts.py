"""Isolated, traceable outputs for one (dataset, config, seed) run.

UPGRADE_PLAN.md M0.5: results must be attributable and parallel-safe. Each run
owns a directory; the archived headline weights are read-only input and are
never a worker's output destination.
"""
import json
import os

import numpy as np

RUN_ROOT = os.path.join('Results', 'runs')

_STATUS = 'status.json'
_PREDICTIONS = 'predictions.npz'
_WEIGHTS = 'weights.npz'


def run_dir(dataset_id: str, config_name: str, seed: int, root: str = RUN_ROOT) -> str:
    """Directory owned exclusively by this run. Created if absent."""
    directory = os.path.join(root, str(dataset_id), str(config_name), 'seed_{}'.format(seed))
    os.makedirs(directory, exist_ok=True)
    return directory


def weights_path(directory: str) -> str:
    return os.path.join(directory, _WEIGHTS)


def _write_status(directory: str, payload: dict) -> None:
    with open(os.path.join(directory, _STATUS), 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _read_status(directory: str) -> dict:
    path = os.path.join(directory, _STATUS)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def start_run(directory: str, config: dict, split_id: str, seed: int, environment: dict) -> None:
    _write_status(directory, {
        'state': 'running',
        'config': config,
        'split_id': split_id,
        'seed': int(seed),
        'environment': environment,
    })


def complete_run(directory: str, metrics: dict) -> None:
    status = _read_status(directory)
    status['state'] = 'complete'
    status['metrics'] = metrics
    _write_status(directory, status)


def fail_run(directory: str, error: str) -> None:
    status = _read_status(directory)
    status['state'] = 'failed'
    status['error'] = str(error)
    _write_status(directory, status)


def is_complete(directory: str) -> bool:
    """True only for a run that finished. Partial output never counts."""
    return _read_status(directory).get('state') == 'complete'


def save_predictions(directory: str, sample_ids, y_true, raw_outputs) -> str:
    """Persist per-example results so paired tests (McNemar) are possible later."""
    path = os.path.join(directory, _PREDICTIONS)
    np.savez(
        path,
        sample_ids=np.asarray(sample_ids),
        y_true=np.asarray(y_true),
        raw_outputs=np.asarray(raw_outputs),
    )
    return path


class TestEvaluationGuard:
    """Permits exactly one test-set evaluation per run (UPGRADE_PLAN.md 0.3).

    Wrap the single final evaluation in ``guard.evaluate(...)``. A second call
    raises, which is what turns "test is evaluated once" from a convention into
    an enforced property.
    """

    def __init__(self):
        self.count = 0

    def evaluate(self, fn, *args, **kwargs):
        if self.count:
            raise RuntimeError(
                'the test set may be evaluated exactly once per run '
                '(UPGRADE_PLAN.md 0.3); this is evaluation number {}'.format(self.count + 1))
        self.count += 1
        return fn(*args, **kwargs)
