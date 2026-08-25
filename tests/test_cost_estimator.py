"""The grid budget gate must be arithmetic, not vibes (UPGRADE_PLAN.md 1.4).

Calibration is measured on the machine and cannot be asserted here; the
projection and the mandated reduction order built on top of it can, and those
are what decide whether a grid is approved.
"""
import json

import numpy as np
import pytest

from QCNN.models.QCNNModel import MAX_BATCHED_AMPLITUDES
from experiments.estimate_cost import (
    DATASET_FLOOR,
    POOLING_ARMS,
    SEED_FLOOR,
    cell_seconds,
    project,
    propose_reduction,
)

CAL = {
    'config': 'proposed', 'n_qubits': 8, 'batch_size': 32, 'epochs': 10,
    'grad_s_per_batch': 0.25,
    'batched_forward_s_per_sample': 0.004,
    'sequential_forward_s_per_sample': 0.13,
    'batched': True,
}


def test_cell_cost_decomposes_into_its_measured_parts():
    cost = cell_seconds(CAL, n_train=343, n_val=86, n_test=142)

    # 343 samples / batch 32 -> 11 batches.
    assert cost['train_s_per_epoch'] == pytest.approx(11 * 0.25)
    # Trainer probes min(200, n_train) training samples plus the whole val split.
    assert cost['monitor_s_per_epoch'] == pytest.approx((200 + 86) * 0.004)
    # The single final test evaluation is priced at the sequential rate.
    assert cost['final_eval_s'] == pytest.approx(142 * 0.13)
    assert cost['total_s'] == pytest.approx(10 * cost['epoch_s'] + cost['final_eval_s'])


def test_the_train_probe_never_exceeds_the_training_set():
    cost = cell_seconds(CAL, n_train=50, n_val=10, n_test=20)
    assert cost['monitor_s_per_epoch'] == pytest.approx((50 + 10) * 0.004)


def test_projection_scales_with_cells_and_divides_by_workers():
    cals = {'proposed': CAL, 'pool_none': dict(CAL, config='pool_none')}
    sizes = {'0v1': (343, 86, 142), '3v5': (343, 86, 142)}

    _, serial_one, _ = project(['proposed'], ['0v1'], [0], cals, sizes, 0.0, 1, False)
    _, serial_many, wall = project(
        list(cals), list(sizes), [0, 1, 2], cals, sizes, 0.0, 4, False)

    # 2 configs x 2 datasets x 3 seeds = 12 cells of identical cost.
    assert serial_many == pytest.approx(serial_one * 12)
    assert wall == pytest.approx(serial_many / 4)


def test_baselines_are_charged_only_to_the_proposed_cell():
    cals = {'proposed': CAL, 'pool_none': dict(CAL, config='pool_none')}
    sizes = {'0v1': (343, 86, 142)}

    _, without, _ = project(list(cals), ['0v1'], [0], cals, sizes, 99.0, 1, False)
    _, with_base, _ = project(list(cals), ['0v1'], [0], cals, sizes, 99.0, 1, True)

    assert with_base - without == pytest.approx(99.0)


def _reduction_setup(n_datasets=4, n_seeds=10):
    cals = {name: dict(CAL, config=name) for name in
            ('proposed', 'pool_none', 'pool_measurement', 'ent_none', 'kernel_ry')}
    names = [f'd{i}' for i in range(n_datasets)]
    sizes = {n: (343, 86, 142) for n in names}
    return cals, names, list(range(n_seeds))


def test_reduction_cuts_seeds_first():
    cals, names, seeds = _reduction_setup()
    _, _, wall = project(list(cals), names, seeds, cals, sizes_of(names), 0.0, 1, False)
    # A budget that only a seed cut can reach.
    budget = wall / 3600.0 * 0.75

    steps, hours = propose_reduction(
        list(cals), names, seeds, cals, sizes_of(names), 0.0, 1, False, budget)

    assert len(steps) == 1 and steps[0].startswith('seeds')
    assert hours <= budget


def test_reduction_never_goes_below_the_seed_floor():
    cals, names, seeds = _reduction_setup()
    steps, _ = propose_reduction(
        list(cals), names, seeds, cals, sizes_of(names), 0.0, 1, False, 0.0001)

    assert f'seeds {len(seeds)} -> {SEED_FLOOR} (floor)' in steps
    assert f'datasets {len(names)} -> {DATASET_FLOOR} (floor)' in steps


def test_reduction_never_cuts_a_pooling_arm():
    """The last resort drops non-pooling ablations only; E1/E2 are not cells."""
    cals, names, seeds = _reduction_setup()
    steps, _ = propose_reduction(
        list(cals), names, seeds, cals, sizes_of(names), 0.0, 1, False, 0.0001)

    dropped = [s for s in steps if s.startswith('drop non-pooling')][0]
    for arm in POOLING_ARMS:
        assert arm not in dropped
    assert 'ent_none' in dropped and 'kernel_ry' in dropped


def sizes_of(names):
    return {n: (343, 86, 142) for n in names}


def _run_cost_main(monkeypatch, tmp_path, calibration, *, with_baselines=False,
                   baseline_failure=None, extra_args=()):
    from experiments import estimate_cost

    output = tmp_path / 'cost.json'
    monkeypatch.setattr(
        estimate_cost, 'prepare_split',
        lambda cfg, pair, dataset_dir, samples: (
            (np.zeros((8, 1)), np.zeros(8), np.zeros((2, 1)), np.zeros(2),
             np.zeros((3, 1)), np.zeros(3)), {}))
    monkeypatch.setattr(estimate_cost, 'calibrate', calibration)
    if baseline_failure is not None:
        monkeypatch.setattr(
            estimate_cost, 'calibrate_baselines',
            lambda *args, **kwargs: (_ for _ in ()).throw(baseline_failure))
    elif with_baselines:
        monkeypatch.setattr(
            estimate_cost, 'calibrate_baselines', lambda *args, **kwargs: 2.0)

    argv = [
        'estimate_cost', '--datasets', '0,1', '--configs', 'proposed',
        '--seeds', '0', '--samples', '10', '--epochs', '1', '--jobs', '1',
        '--output-json', str(output),
    ]
    if not with_baselines:
        argv.append('--no-baselines')
    argv.extend(extra_args)
    monkeypatch.setattr('sys.argv', argv)
    return estimate_cost.main(), json.loads(output.read_text())


def _expected_counts(*, baseline_cells=0, measurable_scheduler=1,
                     measurable_baseline=0, failed=0, total_failures=0):
    return {
        'requested_configs': 1,
        'measurable_configs': int(measurable_scheduler == 1),
        'scheduler_cells': 1,
        'baseline_side_effect_cells': baseline_cells,
        'total_costed_cells': 1 + baseline_cells,
        'measurable_scheduler_cells': measurable_scheduler,
        'measurable_baseline_side_effect_cells': measurable_baseline,
        'measurable_total_costed_cells': measurable_scheduler + measurable_baseline,
        'failed_calibrations': failed,
        'total_failures': total_failures,
    }


@pytest.mark.parametrize('with_baselines, expected', [
    (False, _expected_counts()),
    (True, _expected_counts(baseline_cells=1, measurable_baseline=1)),
])
def test_success_payload_separates_scheduler_and_baseline_cost_cells(
        monkeypatch, tmp_path, with_baselines, expected):
    exit_code, payload = _run_cost_main(
        monkeypatch, tmp_path,
        lambda *args, **kwargs: dict(CAL, epochs=1),
        with_baselines=with_baselines,
    )

    assert exit_code == 0
    assert payload['status'] == 'approved'
    assert payload['counts'] == expected
    assert 'requested_cells' not in payload['counts']
    assert 'measurable_cells' not in payload['counts']


def test_unmeasurable_requested_config_rejects_approval_and_is_recorded(
        monkeypatch, tmp_path):
    def fail_config(*args, **kwargs):
        raise RuntimeError('cannot measure config')

    exit_code, payload = _run_cost_main(monkeypatch, tmp_path, fail_config)

    assert exit_code != 0
    assert payload['status'] == 'failed'
    assert payload['approval']['approved'] is False
    assert payload['counts'] == {
        'requested_configs': 1,
        'measurable_configs': 0,
        'scheduler_cells': 1,
        'baseline_side_effect_cells': 0,
        'total_costed_cells': 1,
        'measurable_scheduler_cells': 0,
        'measurable_baseline_side_effect_cells': 0,
        'measurable_total_costed_cells': 0,
        'failed_calibrations': 1,
        'total_failures': 1,
    }
    assert payload['failures'][0]['kind'] == 'config'
    assert payload['failures'][0]['name'] == 'proposed'
    assert payload['projection'] is None
    assert 'NaN' not in json.dumps(payload) and 'Infinity' not in json.dumps(payload)


def test_unmeasurable_requested_baselines_reject_approval_and_are_recorded(
        monkeypatch, tmp_path):
    exit_code, payload = _run_cost_main(
        monkeypatch, tmp_path, lambda *args, **kwargs: dict(CAL, epochs=1),
        with_baselines=True, baseline_failure=RuntimeError('baseline unsupported'))

    assert exit_code != 0
    assert payload['approval']['approved'] is False
    assert payload['counts'] == {
        'requested_configs': 1,
        'measurable_configs': 1,
        'scheduler_cells': 1,
        'baseline_side_effect_cells': 1,
        'total_costed_cells': 2,
        'measurable_scheduler_cells': 1,
        'measurable_baseline_side_effect_cells': 0,
        'measurable_total_costed_cells': 1,
        'failed_calibrations': 1,
        'total_failures': 1,
    }
    assert payload['failures'][0]['kind'] == 'baseline'
    assert payload['projection'] is None


def test_nonfinite_calibration_is_a_structured_failure(monkeypatch, tmp_path):
    exit_code, payload = _run_cost_main(
        monkeypatch, tmp_path,
        lambda *args, **kwargs: dict(CAL, grad_s_per_batch=float('nan')))

    assert exit_code != 0
    assert payload['status'] == 'failed'
    assert payload['approval']['approved'] is False
    assert payload['projection'] is None
    assert payload['counts'] == _expected_counts(
        measurable_scheduler=0, failed=1, total_failures=1)
    assert payload['failures'][0]['kind'] == 'config'
    assert 'non-finite' in payload['failures'][0]['error']
    assert 'NaN' not in json.dumps(payload) and 'Infinity' not in json.dumps(payload)


def test_json_write_is_atomic_when_serialization_fails(tmp_path):
    from experiments.estimate_cost import _write_json

    output = tmp_path / 'cost.json'
    output.write_text('{"preserved": true}\n')

    with pytest.raises((TypeError, ValueError)):
        _write_json(str(output), {'bad': object()})

    assert json.loads(output.read_text()) == {'preserved': True}


@pytest.mark.parametrize('budget_args', [
    ('--nights', '0'),
    ('--nights', '-1'),
    ('--hours-per-night', '0'),
    ('--hours-per-night', '-1'),
])
def test_nonpositive_budget_is_a_structured_request_failure(
        monkeypatch, tmp_path, budget_args):
    def calibration_must_not_run(*args, **kwargs):
        raise AssertionError('calibration should not run for an invalid budget')

    exit_code, payload = _run_cost_main(
        monkeypatch, tmp_path, calibration_must_not_run, extra_args=budget_args)

    assert exit_code != 0
    assert payload['status'] == 'failed'
    assert payload['approval']['approved'] is False
    assert payload['projection'] is None
    assert payload['counts'] == _expected_counts(
        measurable_scheduler=0, total_failures=1)
    assert payload['failures'][0]['kind'] == 'request'
    assert payload['failures'][0]['name'] == 'budget'


def test_projection_failure_does_not_increment_failed_calibrations(
        monkeypatch, tmp_path):
    from experiments import estimate_cost
    monkeypatch.setattr(
        estimate_cost, 'project',
        lambda *args, **kwargs: ([], 3600.0, float('inf')))

    exit_code, payload = _run_cost_main(
        monkeypatch, tmp_path, lambda *args, **kwargs: dict(CAL, epochs=1))

    assert exit_code != 0
    assert payload['counts'] == _expected_counts(total_failures=1)
    assert payload['failures'][0]['kind'] == 'projection'


def test_memory_cap_forces_the_sequential_path_only_for_large_circuits():
    """The n=16 ablation cannot batch; the n=10 headline comfortably can."""
    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.models.QCNNModel import PureQuantumNativeCNN

    headline = PureQuantumNativeCNN(QuantumNativeConfig.from_image_size(28, 'amplitude'))
    assert headline.num_qubits == 10
    assert headline.supports_batched(32)
    assert headline.max_batch() == MAX_BATCHED_AMPLITUDES // 1024

    wide = PureQuantumNativeCNN(QuantumNativeConfig.from_image_size(4, 'feature_map'))
    assert wide.num_qubits == 16
    assert not wide.supports_batched(32)
    assert wide.max_batch() == MAX_BATCHED_AMPLITUDES // 65536


@pytest.mark.slow
def test_the_fallback_path_agrees_with_the_batched_path(
        headline_model, archived_params, monkeypatch):
    """Falling back to sequential must not change a single output."""
    X = np.random.default_rng(7).random((4, 2 ** 10))
    from QCNN.encoding import PureQuantumEncoder
    X = PureQuantumEncoder.precompute_amplitudes(X, 10)

    batched = np.asarray(
        headline_model.batch_expectations(X, archived_params), dtype=float)

    # Force the cap below this batch so the same call takes the sequential path.
    monkeypatch.setattr(headline_model, 'max_batch', lambda: 1)
    assert not headline_model.supports_batched(len(X))
    fallback = np.asarray(
        headline_model.batch_expectations(X, archived_params), dtype=float)

    np.testing.assert_allclose(fallback, batched, atol=1e-10, rtol=0.0)
