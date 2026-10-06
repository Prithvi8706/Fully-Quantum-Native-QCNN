"""Structural guards for the frozen circuit (UPGRADE_PLAN.md §A2, §A3)."""
import json

import pytest

from QCNN import freeze


def test_circuit_signature_matches_committed_reference(headline_model):
    with open(freeze.SIGNATURE_FIXTURE) as fh:
        reference = json.load(fh)

    signature = freeze.circuit_signature(headline_model)

    assert freeze.signature_hash(signature) == reference["hash"], (
        "The frozen circuit's topology changed. Under UPGRADE_PLAN.md A1 this is "
        "only permitted for a gate proven inert; otherwise revert the change."
    )
    assert signature == reference["signature"]


def test_signature_records_the_known_frozen_topology(headline_model):
    signature = freeze.circuit_signature(headline_model)
    names = [op["name"] for op in signature["operations"]]

    # One amplitude state preparation, one terminal expectation value.
    assert names.count("AmplitudeEmbedding") == 1
    assert names[0] == "AmplitudeEmbedding"
    assert signature["measurements"] == ["expval(Z(0))"]

    # F1/F2: one effective convolution stage and eight pooling pairs at n=10
    # (5 + 2 + 1), with one wire retired unpaired at the 5-active stage.
    assert names.count("CRY") == names.count("CRZ") == 8


def test_only_hard_coded_angle_is_the_inert_discard_rotation(headline_model):
    signature = freeze.circuit_signature(headline_model)
    constants = {
        descriptor
        for op in signature["operations"]
        for descriptor in op["params"]
        if descriptor.startswith("const")
    }
    # F4: the RY(0.02) on already-discarded wires is the sole magic constant.
    # Removing it is governed by the A2 inert-gate exception and is NOT Phase 0 work.
    assert constants == {"const0.02"}


def test_main_path_contains_no_non_unitary_operation(headline_model, archived_params, regression_inputs):
    tape = freeze.headline_tape(headline_model, regression_inputs[0], archived_params)
    assert freeze.unitarity_violations(tape) == []


def test_unitarity_audit_detects_measurement_pooling():
    """Positive control: the audit must flag the measurement-pooling arm.

    'measurement' is the labelled ablation arm from UPGRADE_PLAN.md A3, never
    part of the headline model. If this test stops failing the audit, the audit
    has stopped working.
    """
    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.models.QCNNModel import PureQuantumNativeCNN

    cfg = QuantumNativeConfig.from_image_size(
        freeze.HEADLINE_IMAGE_SIZE, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = 'measurement'
    cfg.device = 'default.qubit'  # lightning.qubit rejects mid-circuit measurement
    arm = PureQuantumNativeCNN(cfg)

    params = arm._flatten_params(arm.quantum_params)
    tape = freeze.headline_tape(arm, freeze.fixed_regression_inputs()[0], params)

    violations = freeze.unitarity_violations(tape)
    # 8 pooling pairs at n=10, each contributing one mid-circuit measurement and
    # the two conditionals Theorem 1's U_1 = RY(gamma).RZ(beta).RY(alpha) requires
    # (the shared RY(gamma) is unconditional). Before the B1 fix on 2026-07-25 the
    # arm emitted a single conditional per pair, which is why it could not tie.
    measurements = [v for v in violations if 'mid-circuit measurement' in v]
    feed_forward = [v for v in violations if 'classical feed-forward' in v]
    assert len(measurements) == 8, violations
    assert len(feed_forward) == 16, violations


def test_headline_has_exactly_one_terminal_measurement(headline_model, archived_params, regression_inputs):
    tape = freeze.headline_tape(headline_model, regression_inputs[0], archived_params)
    assert len(tape.measurements) == 1
