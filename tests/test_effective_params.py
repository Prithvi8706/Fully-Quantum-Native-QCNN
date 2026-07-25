"""Gradient audit of the frozen parameter vector (UPGRADE_PLAN.md 0.2, F1).

269 slots are allocated. Far fewer influence the output. The manuscript must
report both numbers, so both are pinned here.
"""
import json

import numpy as np
import pytest

from QCNN import freeze


@pytest.fixture(scope="module")
def committed_audit():
    with open(freeze.EFFECTIVE_PARAMS_FIXTURE) as fh:
        return json.load(fh)


def test_allocated_slot_count_is_disclosed(committed_audit):
    assert committed_audit["n_allocated"] == freeze.HEADLINE_N_PARAM_SLOTS


def test_effective_count_is_far_below_allocated(committed_audit):
    """F1: the majority of allocated slots receive no gradient."""
    assert committed_audit["n_effective"] < committed_audit["n_allocated"]
    assert committed_audit["n_effective"] == len(committed_audit["effective_slots"])
    assert (
        committed_audit["n_effective"] + len(committed_audit["zero_gradient_slots"])
        == committed_audit["n_allocated"]
    )


def test_gradient_populations_are_separated_by_orders_of_magnitude(committed_audit):
    """The tolerance is justified only if nothing lives near it.

    Gradients split into real signal (>1e-3) and adjoint round-off (~3e-17),
    with an empty band between. The reported effective count therefore does not
    depend on where in that band EFFECTIVE_GRADIENT_TOL is placed.
    """
    grads = np.array(committed_audit["max_abs_gradient"])
    tol = committed_audit["gradient_tolerance"]

    above = grads[grads > tol]
    noise = grads[(grads > 0.0) & (grads <= tol)]

    assert noise.size > 0, "expected some round-off slots; the audit may have changed"
    assert above.min() / noise.max() > 1e10, (
        "signal and round-off populations are no longer cleanly separated; "
        "the tolerance-based effective count needs re-justification"
    )


def test_raw_count_over_reports_relative_to_the_tolerance_count(committed_audit):
    """F1/A6: 'nonzero gradient' is not the same as 'trainable'."""
    assert committed_audit["n_effective_raw"] > committed_audit["n_effective"]


def test_final_readout_rz_receives_no_gradient(committed_audit):
    """Slot 268 is classifier[31], the final RZ on the readout wire.

    RZ commutes with the PauliZ observable, so this gate cannot influence
    <Z_readout>. It is syntactically present but structurally dead, and the
    manuscript must not count it as trainable.
    """
    assert 268 in committed_audit["syntactically_used_slots"]
    assert 268 not in committed_audit["effective_slots"]
    assert committed_audit["max_abs_gradient"][268] == 0.0


def test_dead_pooling_angles_are_the_retired_unpaired_wire(committed_audit):
    """F2 made concrete: the odd-wire retirement costs exactly 3 parameters.

    Layer 0 pools 5 pairs; its pair index 4 is (keep=8, discard=9). At layer 1
    only 5 wires are active, so wire 8 is retired unpaired and never reaches the
    readout. Its three pooling angles (slots 204-206) therefore carry no
    gradient beyond round-off -- which is exactly the round-off population the
    tolerance excludes.
    """
    raw = set(committed_audit["effective_slots_raw"])
    effective = set(committed_audit["effective_slots"])
    pooling_0_start = committed_audit["per_group"]["quantum_pooling_0"]["range"][0]

    round_off = sorted(raw - effective)
    assert round_off == [204, 205, 206]

    # Three consecutive angles, i.e. one whole (CRY, CRZ, RY-on-keep) triple.
    local = [slot - pooling_0_start for slot in round_off]
    assert local == [12, 13, 14]
    assert {index // 3 for index in local} == {4}


def test_only_the_first_convolution_stage_is_effective(committed_audit):
    """F1: layers 1-3 never apply their kernels at n=10 (144 dead slots)."""
    per_group = committed_audit["per_group"]
    assert per_group["quantum_conv_kernel_0"]["n_effective"] == 48
    for layer in (1, 2, 3):
        assert per_group[f"quantum_conv_kernel_{layer}"]["n_effective"] == 0


def test_effective_slots_are_a_subset_of_syntactically_used_slots(committed_audit):
    used = set(committed_audit["syntactically_used_slots"])
    effective = set(committed_audit["effective_slots"])
    assert effective <= used


def test_per_group_counts_sum_to_the_totals(committed_audit):
    per_group = committed_audit["per_group"]
    assert sum(g["n_allocated"] for g in per_group.values()) == committed_audit["n_allocated"]
    assert sum(g["n_effective"] for g in per_group.values()) == committed_audit["n_effective"]


@pytest.mark.slow
def test_audit_reproduces_the_committed_artifact(headline_model, archived_params, regression_inputs, committed_audit):
    fresh = freeze.effective_parameter_audit(headline_model, archived_params, regression_inputs)

    assert fresh["n_allocated"] == committed_audit["n_allocated"]
    assert fresh["n_effective"] == committed_audit["n_effective"]
    assert fresh["effective_slots"] == committed_audit["effective_slots"]
    assert fresh["syntactically_used_slots"] == committed_audit["syntactically_used_slots"]
