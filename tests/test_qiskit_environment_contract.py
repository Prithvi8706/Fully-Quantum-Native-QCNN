"""Contract tests for the isolated, local-only Qiskit environment smoke."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-qiskit-lock.txt"
ENV_ARTIFACT = ROOT / "Results" / "evidence" / "qiskit_environment.json"
TRANSPILE_ARTIFACT = ROOT / "Results" / "evidence" / "qiskit_transpile_smoke.json"
ALLOWED_TRANSPILED_OPERATIONS = {"rz", "sx", "x", "ecr", "measure", "barrier", "reset"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite(nested)


def test_qiskit_dependencies_are_separate_from_frozen_training_lock():
    resolver_input = (ROOT / "requirements-qiskit.in").read_text(encoding="utf-8").splitlines()
    assert resolver_input == [
        "qiskit>=1.4.1,<2",
        "qiskit-aer==0.17.2",
        "qiskit-ibm-runtime==0.40.1",
    ]
    training_lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").lower()
    assert "qiskit" not in training_lock
    assert "pennylane-qiskit" not in training_lock
    qiskit_lock = LOCK.read_text(encoding="utf-8").lower()
    assert "qiskit==" in qiskit_lock
    assert "qiskit-aer==0.17.2" in qiskit_lock
    assert "qiskit-ibm-runtime==0.40.1" in qiskit_lock
    assert "pennylane-qiskit" not in qiskit_lock


def test_environment_artifact_records_isolation_and_local_provenance():
    artifact = _load(ENV_ARTIFACT)
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "pass"
    assert artifact["hardware_environment"]["python_version"].startswith("3.11.")
    assert Path(artifact["hardware_environment"]["executable"]).name.lower() == "python.exe"
    assert artifact["training_environment"]["required_python_version"] == "3.9.13"
    assert artifact["training_environment"]["lock_path"] == "requirements-lock.txt"
    assert artifact["training_environment"]["observed"] is True
    assert artifact["training_environment"]["observed_version"] == "3.9.13"
    assert Path(artifact["training_environment"]["executable"]).name.lower() == "python.exe"
    assert Path(artifact["training_environment"]["executable"]) != Path(
        artifact["hardware_environment"]["executable"]
    )
    assert artifact["network_accessed"] is False
    assert artifact["credentials_accessed"] is False
    assert set(artifact["imports"]) == {"qiskit", "qiskit_aer", "qiskit_ibm_runtime"}
    for module in artifact["imports"].values():
        assert module == {"status": "pass", "exception": None}
    assert set(artifact["installed_versions"]) == {"qiskit", "qiskit-aer", "qiskit-ibm-runtime"}
    assert artifact["lock"]["path"] == "requirements-qiskit-lock.txt"
    assert artifact["lock"]["sha256"] == hashlib.sha256(LOCK.read_bytes()).hexdigest()
    assert artifact["git"]["head"] == re.fullmatch(r"[0-9a-f]{40}", artifact["git"]["head"]).group(0)
    assert isinstance(artifact["git"]["dirty"], bool)
    assert artifact["fake_backend"]["available"] is True
    assert artifact["fake_backend"]["class"]
    assert artifact["fake_backend"]["backend_v2"] is True
    assert artifact["fake_backend"]["target_operations"]
    assert artifact["platform"]
    _assert_finite(artifact)


def test_transpilation_artifact_is_tied_to_canonical_source_and_supported_basis():
    artifact = _load(TRANSPILE_ARTIFACT)
    source = ROOT / artifact["canonical_source"]["path"]
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "pass"
    assert source == ROOT / "QCNN" / "circuits.py"
    assert artifact["canonical_source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact["canonical_source"]["function"] == "build_circuit"
    assert artifact["canonical_source"]["signature_sha256"]
    assert artifact["fixture"]["n_qubits"] in {4, 6}
    assert artifact["fixture"]["parameters"]
    assert artifact["readout"]["measurement_semantics"]
    assert artifact["transpilation"]["seed_transpiler"] == 42
    assert artifact["transpilation"]["optimization_level"] in {0, 1, 2, 3}
    assert artifact["layout"]["policy"]
    assert "initial" in artifact["layout"] and "final" in artifact["layout"]
    assert artifact["backend"]["class"]
    assert artifact["backend"]["target_operations"]
    final_counts = artifact["operations"]["transpiled_counts"]
    assert final_counts
    assert set(final_counts) <= ALLOWED_TRANSPILED_OPERATIONS
    assert artifact["operations"]["unsupported"] == []
    target_validation = artifact["operations"]["target_validation"]
    assert target_validation["uses_actual_target"] is True
    assert target_validation["invalid"] == []
    assert target_validation["checked_instructions"] == sum(final_counts.values()) - final_counts.get(
        "barrier", 0
    )
    for stage in ("logical", "decomposed", "transpiled"):
        assert set(artifact["metrics"][stage]) == {"depth", "width", "size", "operation_counts"}
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["payload_sha256"])
    assert artifact["equivalence"]["result"] in {"pass", "limited"}
    assert artifact["equivalence"]["method"]
    assert artifact["equivalence"]["limitations"]
    assert artifact["network_accessed"] is False
    assert artifact["credentials_accessed"] is False
    _assert_finite(artifact)


def test_documentation_preserves_training_environment_and_gates_real_qpu_use():
    docs = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "RUN_GUIDE.md")
    )
    for required in (
        "Python 3.9.13",
        "requirements-lock.txt",
        "Python 3.11",
        ".venv-qiskit",
        "requirements-qiskit-lock.txt",
        "scripts/check_qiskit_environment.py",
        "no credentials or network",
        "nothing in this repository runs on quantum hardware",
    ):
        assert required.lower() in docs.lower()
