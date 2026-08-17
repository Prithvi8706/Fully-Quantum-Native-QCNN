"""Task 4 guards for headline-geometry ablations and safe reuse."""
import json

from QCNN import freeze
from QCNN.utils import run_artifacts
from experiments import run_experiments as runner


GENERAL_ARMS = (
    "proposed",
    "pool_none",
    "pool_measurement",
    "ent_one_diagonal",
    "ent_none",
    "kernel_ry",
)
EXPECTED_FACTOR = {
    "pool_none": ("pooling_mode", "none"),
    "pool_measurement": ("pooling_mode", "measurement"),
    "ent_one_diagonal": ("conv_entanglement", "one_diagonal"),
    "ent_none": ("conv_entanglement", "none"),
    "kernel_ry": ("kernel_rotations", "ry"),
}


def test_general_amplitude_arms_resolve_to_headline_geometry():
    for name in GENERAL_ARMS:
        resolved = runner.resolve_ablation(name)
        assert resolved["image_size"] == 28
        assert resolved["n_qubits"] == freeze.HEADLINE_N_QUBITS == 10
        assert resolved["encoding_type"] == "amplitude"
        assert "geometry_exception" not in resolved


def test_general_arms_change_exactly_the_named_factor():
    reference = runner.resolve_ablation("proposed")
    factors = ("pooling_mode", "conv_entanglement", "kernel_rotations")

    for name, (changed_factor, changed_value) in EXPECTED_FACTOR.items():
        resolved = runner.resolve_ablation(name)
        differences = [factor for factor in factors
                       if resolved[factor] != reference[factor]]
        assert differences == [changed_factor]
        assert resolved[changed_factor] == changed_value
        assert resolved["factor"] == changed_factor


def test_feature_map_is_the_sole_structured_geometry_exception():
    exceptions = runner.ABLATION_GEOMETRY_EXCEPTIONS
    assert set(exceptions) == {"enc_feature_map"}

    resolved = runner.resolve_ablation("enc_feature_map")
    exception = resolved["geometry_exception"]
    assert exception == exceptions["enc_feature_map"]
    assert exception == {
        "code": "encoding_requires_distinct_geometry",
        "factor": "encoding_type",
        "reference_image_size": 28,
        "reference_n_qubits": 10,
        "image_size": 4,
        "n_qubits": 16,
        "reason": "Feature-map simulation uses one qubit per pixel and is not feasible at headline geometry.",
        "comparable_as_one_factor": False,
    }


def test_e3_pooling_arms_remain_at_headline_geometry():
    for name in runner.E3_POOLING_ARMS:
        resolved = runner.resolve_ablation(name)
        assert resolved["image_size"] == 28
        assert resolved["n_qubits"] == 10
        assert "geometry_exception" not in resolved


def test_old_n8_ablation_cell_is_not_reusable(tmp_path, monkeypatch):
    pair = (0, 1)
    name = "proposed"
    seed = 3
    run_root = tmp_path / "runs"
    exp_root = tmp_path / "experiments"
    monkeypatch.setattr(run_artifacts, "RUN_ROOT", str(run_root))
    monkeypatch.setattr(runner, "EXP_ROOT", str(exp_root))

    metrics_path = exp_root / "0v1" / name / f"seed_{seed}.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({"accuracy": 0.9}))

    old_config = runner._expected_config(name, seed, epochs=30)
    old_config.update(image_size=16, n_qubits=8, n_features=256)
    directory = run_artifacts.run_dir("0v1", name, seed)
    run_artifacts.start_run(
        directory, config=old_config, split_id="old-split", seed=seed,
        environment={"python": "3.9.13"})
    run_artifacts.save_predictions(directory, [0], [1], [0.5])
    import numpy as np
    np.savez(run_artifacts.weights_path(directory), w=np.zeros(1))
    run_artifacts.complete_run(directory, metrics={"accuracy": 0.9})

    assert not runner._is_reusable_cell(pair, name, seed, epochs=30)
