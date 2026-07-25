"""The grid budget gate must be arithmetic, not vibes (UPGRADE_PLAN.md 1.4).

Calibration is measured on the machine and cannot be asserted here; the
projection and the mandated reduction order built on top of it can, and those
are what decide whether a grid is approved.
"""
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
