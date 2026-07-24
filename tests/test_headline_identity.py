"""The archived headline model must be reconstructible from committed artifacts.

Everything downstream in the Q1 upgrade is defined relative to this exact model
(UPGRADE_PLAN.md M0.1), so its identity is asserted before anything else.
"""
import json
import os

import numpy as np
import pytest

from QCNN import freeze


def test_headline_config_matches_archived_metadata():
    with open(os.path.join("Results", "metadata.json")) as fh:
        archived = json.load(fh)["config"]

    model = freeze.build_headline_model()
    cfg = model.config

    assert cfg.image_size == archived["image_size"] == freeze.HEADLINE_IMAGE_SIZE
    assert cfg.n_qubits == archived["n_qubits"] == freeze.HEADLINE_N_QUBITS
    assert cfg.n_features == archived["n_features"]
    assert cfg.encoding_type == archived["encoding_type"] == freeze.HEADLINE_ENCODING
    assert cfg.n_conv_layers == archived["n_conv_layers"]
    assert cfg.device == archived["device"]
    assert cfg.shots is archived["shots"] is None


def test_headline_allocates_269_parameter_slots():
    model = freeze.build_headline_model()
    flat = model._flatten_params(model.quantum_params)
    assert flat.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)


def test_archived_weights_load_into_the_headline_layout():
    model = freeze.build_headline_model()
    params = freeze.load_archived_params(model)
    assert params.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)
    assert np.all(np.isfinite(np.array(params)))


def test_slot_ranges_tile_the_parameter_vector():
    model = freeze.build_headline_model()
    ranges = freeze.slot_ranges(model)
    assert list(ranges) == [
        "quantum_conv_kernel_0", "quantum_conv_kernel_1",
        "quantum_conv_kernel_2", "quantum_conv_kernel_3",
        "quantum_pooling_0", "quantum_pooling_1", "quantum_pooling_2",
        "quantum_classifier",
    ]
    cursor = 0
    for start, stop in ranges.values():
        assert start == cursor
        cursor = stop
    assert cursor == freeze.HEADLINE_N_PARAM_SLOTS


def test_regression_inputs_are_fixed_and_reproducible():
    first = freeze.fixed_regression_inputs()
    second = freeze.fixed_regression_inputs()
    assert first.shape == (freeze.N_REGRESSION_INPUTS, 2 ** freeze.HEADLINE_N_QUBITS)
    np.testing.assert_array_equal(first, second)
