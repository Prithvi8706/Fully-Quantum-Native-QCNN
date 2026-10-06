"""Headline-size schedule and execution contract for quantum baselines."""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from baselines import quantum_baselines


_EXPECTED_SCHEDULES = {
    4: [
        {
            "input_wires": [0, 1, 2, 3],
            "pairs": [{"keep": 0, "discard": 1}, {"keep": 2, "discard": 3}],
            "unpaired": [],
            "output_wires": [0, 2],
        },
        {
            "input_wires": [0, 2],
            "pairs": [{"keep": 0, "discard": 2}],
            "unpaired": [],
            "output_wires": [0],
        },
    ],
    6: [
        {
            "input_wires": [0, 1, 2, 3, 4, 5],
            "pairs": [
                {"keep": 0, "discard": 1},
                {"keep": 2, "discard": 3},
                {"keep": 4, "discard": 5},
            ],
            "unpaired": [],
            "output_wires": [0, 2, 4],
        },
        {
            "input_wires": [0, 2, 4],
            "pairs": [{"keep": 0, "discard": 2}],
            "unpaired": [{"wire": 4, "treatment": "retired_without_operation"}],
            "output_wires": [0],
        },
    ],
    8: [
        {
            "input_wires": list(range(8)),
            "pairs": [
                {"keep": 0, "discard": 1},
                {"keep": 2, "discard": 3},
                {"keep": 4, "discard": 5},
                {"keep": 6, "discard": 7},
            ],
            "unpaired": [],
            "output_wires": [0, 2, 4, 6],
        },
        {
            "input_wires": [0, 2, 4, 6],
            "pairs": [{"keep": 0, "discard": 2}, {"keep": 4, "discard": 6}],
            "unpaired": [],
            "output_wires": [0, 4],
        },
        {
            "input_wires": [0, 4],
            "pairs": [{"keep": 0, "discard": 4}],
            "unpaired": [],
            "output_wires": [0],
        },
    ],
    10: [
        {
            "input_wires": list(range(10)),
            "pairs": [
                {"keep": 0, "discard": 1},
                {"keep": 2, "discard": 3},
                {"keep": 4, "discard": 5},
                {"keep": 6, "discard": 7},
                {"keep": 8, "discard": 9},
            ],
            "unpaired": [],
            "output_wires": [0, 2, 4, 6, 8],
        },
        {
            "input_wires": [0, 2, 4, 6, 8],
            "pairs": [{"keep": 0, "discard": 2}, {"keep": 4, "discard": 6}],
            "unpaired": [{"wire": 8, "treatment": "retired_without_operation"}],
            "output_wires": [0, 4],
        },
        {
            "input_wires": [0, 4],
            "pairs": [{"keep": 0, "discard": 4}],
            "unpaired": [],
            "output_wires": [0],
        },
    ],
}


@pytest.mark.parametrize("n_qubits", [4, 6, 8, 10])
def test_active_wire_schedule_is_exact_and_terminates_at_one_wire(n_qubits):
    schedule = quantum_baselines.active_wire_schedule(n_qubits)

    assert schedule == _EXPECTED_SCHEDULES[n_qubits]
    assert schedule[-1]["output_wires"] == [0]


@pytest.mark.parametrize("n_qubits", [4, 6, 8, 10])
def test_pairs_are_disjoint_and_odd_wires_are_explicit(n_qubits):
    for stage in quantum_baselines.active_wire_schedule(n_qubits):
        paired = [wire for pair in stage["pairs"] for wire in pair.values()]
        unpaired = [entry["wire"] for entry in stage["unpaired"]]

        assert len(paired) == len(set(paired))
        assert sorted(paired + unpaired) == stage["input_wires"]
        assert all(entry["treatment"] == "retired_without_operation"
                   for entry in stage["unpaired"])


@pytest.mark.parametrize("n_qubits", [0, 1])
def test_active_wire_schedule_rejects_fewer_than_two_qubits(n_qubits):
    with pytest.raises(ValueError, match="at least 2"):
        quantum_baselines.active_wire_schedule(n_qubits)


@pytest.mark.parametrize(
    ("n_qubits", "expected"),
    [
        (4, {"cong": 18, "hur": 22, "ttn": 18}),
        (6, {"cong": 18, "hur": 22, "ttn": 24}),
        (8, {"cong": 27, "hur": 33, "ttn": 42}),
        (10, {"cong": 27, "hur": 33, "ttn": 48}),
    ],
)
def test_parameter_allocation_and_consumption_match_schedule(n_qubits, expected):
    for name, architecture in quantum_baselines._ARCHITECTURES.items():
        count = architecture["param_count"](n_qubits)
        cursor = quantum_baselines._Cursor(pnp.zeros(count))

        qml.tape.make_qscript(lambda: architecture["body"](cursor, n_qubits))()

        assert count == expected[name]
        assert cursor.i == count


def _tiny_split(n_qubits=10):
    basis = np.eye(2 ** n_qubits)
    return {
        "X_train": basis[[0, 1]],
        "y_train": np.array([-1.0, 1.0]),
        "X_val": basis[[2, 3]],
        "y_val": np.array([-1.0, 1.0]),
        "X_test": basis[[4, 5]],
        "y_test": np.array([-1.0, 1.0]),
    }


def test_n10_quantum_baselines_train_one_epoch_and_evaluate_test_once():
    results = quantum_baselines.run_quantum_baselines(
        **_tiny_split(), seed=7, n_qubits=10, n_epochs=1, learning_rate=0.01)

    assert set(results) == {"cong", "hur", "ttn"}
    for result in results.values():
        assert result["selection"]["best_epoch"] == 1
        assert result["selection"]["n_validation_evaluations"] == 1
        assert result["test_evaluations"] == 1
        assert result["raw_outputs"].shape == (2,)


@pytest.mark.parametrize("arch_name", ["cong", "hur", "ttn"])
def test_n10_batched_and_sequential_predictions_agree(arch_name):
    architecture = quantum_baselines._ARCHITECTURES[arch_name]
    n_params = architecture["param_count"](10)
    params = pnp.linspace(-0.2, 0.2, n_params)
    samples = _tiny_split()["X_test"]
    device = qml.device("default.qubit", wires=10)

    @qml.qnode(device)
    def circuit(x):
        quantum_baselines._amp_encode(x, 10)
        readout = architecture["body"](quantum_baselines._Cursor(params), 10)
        return qml.expval(qml.PauliZ(readout))

    batched = np.asarray(circuit(samples))
    sequential = np.asarray([circuit(sample) for sample in samples])

    np.testing.assert_allclose(batched, sequential, atol=1e-9)
