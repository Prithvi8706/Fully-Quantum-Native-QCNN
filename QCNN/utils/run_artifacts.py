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


def run_dir(dataset_id: str, config_name: str, seed: int, root: str = None,
            create: bool = True) -> str:
    """Directory owned exclusively by this run.

    ``create=False`` returns the path without touching the filesystem, so a
    resume check cannot leave empty directories behind for cells it skips.
    ``root`` resolves at call time so tests can redirect the whole tree.
    """
    root = RUN_ROOT if root is None else root
    directory = os.path.join(root, str(dataset_id), str(config_name), 'seed_{}'.format(seed))
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory


def weights_path(directory: str) -> str:
    return os.path.join(directory, _WEIGHTS)


def _write_status(directory: str, payload: dict) -> None:
    with open(os.path.join(directory, _STATUS), 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _read_status(directory: str) -> dict:
    path = os.path.join(directory, _STATUS)
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


# Every key start_run/complete_run is contracted to write. A status file missing
# any of them predates the current schema and cannot be trusted for resume.
_REQUIRED_STATUS_KEYS = ('state', 'config', 'split_id', 'seed', 'environment', 'metrics')


def _comparable_config(config: dict) -> dict:
    """Config identity for resume, minus fields that are recorded separately.

    ``split_id`` is stamped onto the config object by the runner *during* the
    run, so a pre-run preview of the same cell cannot carry it; the status file
    records it as a top-level field regardless.
    """
    if not isinstance(config, dict):
        raise TypeError('config must be an object')
    return {k: v for k, v in config.items() if k != 'split_id'}


def _same_config_identity(recorded, expected) -> bool:
    if isinstance(recorded, dict) and isinstance(expected, dict):
        return (
            recorded.keys() == expected.keys()
            and all(_same_config_identity(recorded[key], expected[key])
                    for key in recorded)
        )
    if isinstance(recorded, list) and isinstance(expected, list):
        return (
            len(recorded) == len(expected)
            and all(_same_config_identity(left, right)
                    for left, right in zip(recorded, expected))
        )
    numeric_types = (bool, int, float)
    if isinstance(recorded, numeric_types) or isinstance(expected, numeric_types):
        return type(recorded) is type(expected) and recorded == expected
    return recorded == expected


def _valid_npz_artifacts(directory: str) -> bool:
    try:
        with np.load(os.path.join(directory, _WEIGHTS), allow_pickle=False) as weights:
            if not weights.files:
                return False
            has_parameter_value = False
            for name in weights.files:
                values = np.asarray(weights[name])
                if values.size == 0:
                    continue
                has_parameter_value = True
                if not np.all(np.isfinite(values)):
                    return False
            if not has_parameter_value:
                return False
        with np.load(os.path.join(directory, _PREDICTIONS), allow_pickle=False) as predictions:
            required = ('sample_ids', 'y_true', 'raw_outputs')
            if any(name not in predictions.files for name in required):
                return False
            arrays = [np.asarray(predictions[name]) for name in required]
            if any(values.ndim != 1 for values in arrays):
                return False
            if len({len(values) for values in arrays}) != 1:
                return False
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return False
    return True


def is_reusable(directory: str, config: dict = None, seed: int = None) -> bool:
    """True only for a complete, schema-valid, identity-matching run.

    UPGRADE_PLAN.md 1.3 / roadmap M1.3: a resumed sweep must skip *only* cells
    it would otherwise reproduce exactly. Anything partial, anything written by
    an older schema, and anything whose config or seed has since changed is
    re-run rather than silently reused.
    """
    status = _read_status(directory)
    if status.get('state') != 'complete':
        return False
    if any(key not in status for key in _REQUIRED_STATUS_KEYS):
        return False
    if not isinstance(status.get('metrics'), dict) or not status['metrics']:
        return False
    if not _valid_npz_artifacts(directory):
        return False
    try:
        if type(status['seed']) is not int:
            return False
        if seed is not None and (type(seed) is not int or status['seed'] != seed):
            return False
        if config is not None:
            if not _same_config_identity(
                    _comparable_config(status['config']),
                    _comparable_config(config)):
                return False
    except (TypeError, ValueError, AttributeError):
        return False
    return True


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


def save_baseline_result(*, directory: str, result: dict, split_id: str,
                         sample_ids, y_test, seed: int, environment: dict) -> None:
    """Atomically complete the provenance contract for one baseline model."""
    start_run(directory, config={"artifact_schema_version": 1}, split_id=split_id,
              seed=seed, environment=environment)
    selection = dict(result["selection"])
    selection["test_evaluations"] = int(result["test_evaluations"])
    with open(os.path.join(directory, "selection.json"), "w") as fh:
        json.dump(selection, fh, indent=2, sort_keys=True)
    metrics = {k: v for k, v in result["metrics"].items()
               if isinstance(v, (int, float, str, bool, list, type(None)))}
    with open(os.path.join(directory, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    selected_parameters = np.asarray(result.get("selected_parameters", []), dtype=float)
    np.savez(weights_path(directory), selected_parameters=selected_parameters)
    save_predictions(directory, sample_ids, y_test, result["raw_outputs"])
    complete_run(directory, metrics=metrics)


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
