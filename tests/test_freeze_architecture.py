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
