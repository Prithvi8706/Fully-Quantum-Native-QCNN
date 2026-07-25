"""Resume must skip only cells it would otherwise reproduce exactly
(UPGRADE_PLAN.md 1.3, roadmap M1.3).

The dangerous failure here is silent: a sweep that reuses a run produced under a
different config, a different seed, or a half-written directory, and reports the
stale number as if it were fresh. Every rejection path below exists to make that
impossible.
"""
import json
import os

import numpy as np
import pytest

from QCNN.utils import run_artifacts

CONFIG = {'n_epochs': 30, 'pooling_mode': 'unitary', 'n_qubits': 8, 'seed': 3}


def _complete_run(directory, config=None, seed=3):
    """A directory that satisfies every reuse condition."""
    run_artifacts.start_run(
        directory, config=dict(config if config is not None else CONFIG),
        split_id='split-abc', seed=seed, environment={'python': '3.9.13'})
    run_artifacts.save_predictions(directory, [0, 1], [1, -1], [0.4, -0.6])
    np.savez(run_artifacts.weights_path(directory), w=np.zeros(3))
    run_artifacts.complete_run(directory, metrics={'accuracy': 0.97})
    return directory


def test_a_complete_matching_run_is_reusable(tmp_path):
    d = _complete_run(str(tmp_path))
    assert run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_an_unfinished_run_is_not_reusable(tmp_path):
    d = str(tmp_path)
    run_artifacts.start_run(d, config=dict(CONFIG), split_id='split-abc', seed=3,
                            environment={'python': '3.9.13'})
    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_a_failed_run_is_not_reusable(tmp_path):
    d = _complete_run(str(tmp_path))
    run_artifacts.fail_run(d, error='boom')
    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=3)


@pytest.mark.parametrize("missing", ['weights.npz', 'predictions.npz'])
def test_a_run_missing_its_artifacts_is_not_reusable(tmp_path, missing):
    d = _complete_run(str(tmp_path))
    os.remove(os.path.join(d, missing))
    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_a_different_seed_is_not_reusable(tmp_path):
    d = _complete_run(str(tmp_path), seed=3)
    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=4)


def test_a_different_config_is_not_reusable(tmp_path):
    d = _complete_run(str(tmp_path))
    changed = dict(CONFIG, n_epochs=50)
    assert not run_artifacts.is_reusable(d, config=changed, seed=3)


def test_an_ablation_switch_is_not_reusable(tmp_path):
    """The case that would quietly put unitary-pooling numbers in a pool_none row."""
    d = _complete_run(str(tmp_path))
    assert not run_artifacts.is_reusable(
        d, config=dict(CONFIG, pooling_mode='none'), seed=3)


def test_split_id_alone_does_not_block_reuse(tmp_path):
    """split_id is stamped during the run, so a pre-run preview cannot carry it."""
    d = _complete_run(str(tmp_path), config=dict(CONFIG, split_id='split-abc'))
    assert run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_empty_metrics_are_not_reusable(tmp_path):
    d = _complete_run(str(tmp_path))
    run_artifacts.complete_run(d, metrics={})
    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_an_older_schema_is_not_reusable(tmp_path):
    """A status file predating the current contract is re-run, not trusted."""
    d = str(tmp_path)
    _complete_run(d)
    path = os.path.join(d, 'status.json')
    with open(path) as fh:
        status = json.load(fh)
    del status['environment']
    with open(path, 'w') as fh:
        json.dump(status, fh)

    assert not run_artifacts.is_reusable(d, config=CONFIG, seed=3)


def test_run_dir_can_be_resolved_without_creating_it(tmp_path):
    """A resume check must not litter empty directories for cells it skips."""
    path = run_artifacts.run_dir('0v1', 'proposed', 7, root=str(tmp_path), create=False)
    assert not os.path.exists(path)

    assert run_artifacts.run_dir('0v1', 'proposed', 7, root=str(tmp_path)) == path
    assert os.path.isdir(path)


MNIST_DIR = os.path.join('datasets', 'MNIST')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(MNIST_DIR), reason='MNIST not present')
def test_repeated_smoke_resumes_and_reproduces_the_summary(tmp_path, monkeypatch, capsys):
    """The M1.3 exit check, end to end, against isolated output roots.

    Runs the smoke sweep twice. The second pass must re-run nothing and produce
    a byte-identical summary.csv.
    """
    from experiments import run_experiments as runner

    exp_root = str(tmp_path / 'experiments')
    run_root = str(tmp_path / 'runs')
    monkeypatch.setattr(runner, 'EXP_ROOT', exp_root)
    monkeypatch.setattr(runner, 'FAILURE_MANIFEST', os.path.join(exp_root, 'failures.json'))
    monkeypatch.setattr(run_artifacts, 'RUN_ROOT', run_root)
    monkeypatch.setattr(runner, 'MANIFEST_ROOT', str(tmp_path / 'manifests'))

    argv = ['run_experiments', '--quick', '--no-baselines', '--mnist-dir', MNIST_DIR]
    monkeypatch.setattr('sys.argv', argv)

    runner.main()
    first = capsys.readouterr().out
    with open(os.path.join(exp_root, 'summary.csv')) as fh:
        summary_first = fh.read()

    assert '4 cells to run, 0 reused' in first
    with open(os.path.join(exp_root, 'failures.json')) as fh:
        assert json.load(fh)['n_failed'] == 0

    runner.main()
    second = capsys.readouterr().out
    with open(os.path.join(exp_root, 'summary.csv')) as fh:
        summary_second = fh.read()

    assert '0 cells to run, 4 reused' in second
    assert summary_second == summary_first


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(MNIST_DIR), reason='MNIST not present')
def test_a_changed_epoch_budget_defeats_resume(tmp_path, monkeypatch, capsys):
    """Reuse is identity-matched, not merely presence-matched."""
    from experiments import run_experiments as runner

    exp_root = str(tmp_path / 'experiments')
    monkeypatch.setattr(runner, 'EXP_ROOT', exp_root)
    monkeypatch.setattr(runner, 'FAILURE_MANIFEST', os.path.join(exp_root, 'failures.json'))
    monkeypatch.setattr(run_artifacts, 'RUN_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr(runner, 'MANIFEST_ROOT', str(tmp_path / 'manifests'))

    # Not --quick: that flag overrides --epochs after parsing, so the budget
    # has to be set explicitly for this comparison to mean anything.
    base = ['run_experiments', '--no-baselines', '--mnist-dir', MNIST_DIR,
            '--datasets', '0,1', '--configs', 'proposed', '--seeds', '0',
            '--samples', '40']
    monkeypatch.setattr('sys.argv', base + ['--epochs', '2'])
    runner.main()
    assert '1 cells to run, 0 reused' in capsys.readouterr().out

    # Identical command: reused.
    monkeypatch.setattr('sys.argv', base + ['--epochs', '2'])
    runner.main()
    assert '0 cells to run, 1 reused' in capsys.readouterr().out

    # Same cell, different epoch budget: must be re-run, not reused.
    monkeypatch.setattr('sys.argv', base + ['--epochs', '3'])
    runner.main()
    assert '1 cells to run, 0 reused' in capsys.readouterr().out


def test_failure_manifest_is_written_even_when_clean(tmp_path, monkeypatch):
    from experiments import run_experiments as runner
    monkeypatch.setattr(runner, 'EXP_ROOT', str(tmp_path))
    monkeypatch.setattr(runner, 'FAILURE_MANIFEST', os.path.join(str(tmp_path), 'failures.json'))

    runner._write_failure_manifest([])
    with open(runner.FAILURE_MANIFEST) as fh:
        assert json.load(fh) == {'n_failed': 0, 'failures': []}

    runner._write_failure_manifest([{'dataset': '0v1', 'config': 'proposed',
                                     'seed': 0, 'error': 'boom'}])
    with open(runner.FAILURE_MANIFEST) as fh:
        payload = json.load(fh)
    assert payload['n_failed'] == 1
    assert payload['failures'][0]['config'] == 'proposed'
